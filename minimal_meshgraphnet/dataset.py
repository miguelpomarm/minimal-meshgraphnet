# pylint: disable=g-bad-file-header
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ============================================================================
# Modifications by Miguel Pomar Martinez (2026):
#   - The 'train' split is loaded from multiple TFRecord chunks
#     (train_v2_chunk_0..N.tfrecord) produced by the angular preprocessing
#     script, which precomputes edge_bin_start and edge_bin_width via the
#     Sinkhorn-Knopp algorithm.
#   - The 'test' split reads test_v2.tfrecord (also preprocessed).
#   - Parser and shuffle logic are unchanged: the buffer of 10000 mixes
#     frames across chunks regardless of chunk order.
# Bachelor's Thesis - Universidad de Zaragoza (EINA) - 2026
# Reference: T. Pfaff et al., "Learning Mesh-Based Simulation with Graph
# Networks", ICLR 2021. arXiv:2010.03409
# ============================================================================
"""Utility functions for reading the datasets."""

import functools
import json
import os

import tensorflow.compat.v1 as tf

from minimal_meshgraphnet.common import NodeType


# Number of chunks in which the preprocessed 'train' split is stored.
NUM_TRAIN_CHUNKS = 4


def _parse(proto, meta):
  """Parse a trajectory from a tf.Example."""
  feature_lists = {k: tf.io.VarLenFeature(tf.string)
                   for k in meta['field_names']}
  features = tf.io.parse_single_example(proto, feature_lists)
  out = {}
  for key, field in meta['features'].items():
    data = tf.io.decode_raw(features[key].values, getattr(tf, field['dtype']))
    data = tf.reshape(data, field['shape'])
    if field['type'] == 'static':
      data = tf.tile(data, [meta['trajectory_length'], 1, 1])
    elif field['type'] == 'dynamic_varlen':
      length = tf.io.decode_raw(features['length_'+key].values, tf.int32)
      length = tf.reshape(length, [-1])
      data = tf.RaggedTensor.from_row_lengths(data, row_lengths=length)
    elif field['type'] != 'dynamic':
      raise ValueError('invalid data format')
    out[key] = data
  return out


def load_dataset(path, split):
  """Load a dataset split.

  For the 'train' split, the dataset is loaded from a list of preprocessed
  TFRecord chunks (train_v2_chunk_0..N-1.tfrecord). Each chunk contains
  trajectories with the precomputed angular bin ranges (edge_bin_start,
  edge_bin_width) needed by the angular scatter operation.

  For the other splits (test, valid), a single TFRecord file is used.
  """
  with tf.io.gfile.GFile(os.path.join(path, 'meta.json'), 'r') as fp:
    meta = json.loads(fp.read())

  if split == 'train':
    # Load the 4 preprocessed chunks as a list of TFRecord files
    files = [os.path.join(path, 'train_v2_chunk_{}.tfrecord'.format(i))
             for i in range(NUM_TRAIN_CHUNKS)]
  else:
    files = os.path.join(
        path,
        ('test_v2.tfrecord' if split == 'test' else split + '.tfrecord'))

  ds = tf.data.TFRecordDataset(files)
  ds = ds.map(functools.partial(_parse, meta=meta), num_parallel_calls=8)
  ds = ds.prefetch(1)
  return ds


def add_targets(ds, fields, add_history):
  """Add target and (optionally) history fields to the dataset."""
  def fn(trajectory):
    out = {}
    for key, val in trajectory.items():
      out[key] = val[1:-1]
      if key in fields:
        if add_history:
          out['prev|'+key] = val[0:-2]
        out['target|'+key] = val[2:]
    return out
  return ds.map(fn, num_parallel_calls=8)


def split_and_preprocess(ds, noise_field, noise_scale, noise_gamma):
  """Split trajectories into individual frames and add training noise."""
  def add_noise(frame):
    noise = tf.random.normal(tf.shape(frame[noise_field]),
                             stddev=noise_scale, dtype=tf.float32)
    # Do not apply noise to boundary nodes
    mask = tf.equal(frame['node_type'], NodeType.NORMAL)[:, 0]
    noise = tf.where(mask, noise, tf.zeros_like(noise))
    frame[noise_field] += noise
    frame['target|'+noise_field] += (1.0 - noise_gamma) * noise
    return frame

  ds = ds.flat_map(tf.data.Dataset.from_tensor_slices)
  ds = ds.map(add_noise, num_parallel_calls=8)
  ds = ds.shuffle(10000)
  ds = ds.repeat(None)
  return ds.prefetch(10)


def batch_dataset(ds, batch_size):
  """Batch a dataset by concatenating frames within a window."""
  shapes = ds.output_shapes
  types = ds.output_types
  def renumber(buffer, frame):
    nodes, cells = buffer
    new_nodes, new_cells = frame
    return nodes + new_nodes, tf.concat([cells, new_cells+nodes], axis=0)

  def batch_accumulate(ds_window):
    out = {}
    for key, ds_val in ds_window.items():
      initial = tf.zeros((0, shapes[key][1]), dtype=types[key])
      if key == 'cells':
        # Renumber node indices in cells to account for the concatenation
        num_nodes = ds_window['node_type'].map(lambda x: tf.shape(x)[0])
        cells = tf.data.Dataset.zip((num_nodes, ds_val))
        initial = (tf.constant(0, tf.int32), initial)
        _, out[key] = cells.reduce(initial, renumber)
      else:
        merge = lambda prev, cur: tf.concat([prev, cur], axis=0)
        out[key] = ds_val.reduce(initial, merge)
    return out

  if batch_size > 1:
    ds = ds.window(batch_size, drop_remainder=True)
    ds = ds.map(batch_accumulate, num_parallel_calls=8)
  return ds