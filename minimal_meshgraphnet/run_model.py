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
#   - Warmup schedule for the first 1000 steps (global_step increments but
#     the optimiser is not applied), to stabilise the normalisers.
#   - Automatic logging of the training loss to a CSV file every 100 steps.
# Bachelor's Thesis - Universidad de Zaragoza (EINA) - 2026
# Reference: T. Pfaff et al., "Learning Mesh-Based Simulation with Graph
# Networks", ICLR 2021. arXiv:2010.03409
# ============================================================================
"""CLI entry point for training and evaluating the learned model."""

import pickle
import os

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf

# ---------------------------------------------------------------------------
# NOTE ON CLOTH: cloth_model and cloth_eval are kept here for reference to
# the original DeepMind implementation, but running this pipeline with
# --model=cloth will NOT work out of the box. The proposed angular
# reinterpretation of the latent space requires each dataset to be
# preprocessed with the Sinkhorn-Knopp step, which computes the angular
# bin ranges (edge_bin_start, edge_bin_width) for every real edge.
# The CylinderFlow dataset is provided in preprocessed form; a similar
# preprocessing script would be needed to adapt cloth.
# ---------------------------------------------------------------------------
from minimal_meshgraphnet import cfd_eval
from minimal_meshgraphnet import cfd_model
from minimal_meshgraphnet import cloth_eval
from minimal_meshgraphnet import cloth_model
from minimal_meshgraphnet import core_model
from minimal_meshgraphnet import dataset

FLAGS = flags.FLAGS
flags.DEFINE_enum('mode', 'train', ['train', 'eval'],
                  'Train model, or run evaluation.')
flags.DEFINE_enum('model', None, ['cfd', 'cloth'], 'Select model to run.')
flags.DEFINE_string('checkpoint_dir', None, 'Directory to save checkpoint')
flags.DEFINE_string('dataset_dir', None, 'Directory to load dataset from.')
flags.DEFINE_string('rollout_path', None,
                    'Pickle file to save eval trajectories')
flags.DEFINE_enum('rollout_split', 'valid', ['train', 'test', 'valid'],
                  'Dataset split to use for rollouts.')
flags.DEFINE_integer('num_rollouts', 10, 'No. of rollout trajectories')
flags.DEFINE_integer('num_training_steps', int(10e6), 'No. of training steps')

PARAMETERS = {
    'cfd': dict(noise=0.02, gamma=1.0, field='velocity', history=False,
                size=2, batch=2, model=cfd_model, evaluator=cfd_eval),
    'cloth': dict(noise=0.003, gamma=0.1, field='world_pos', history=True,
                  size=3, batch=1, model=cloth_model, evaluator=cloth_eval)
}


def learner(model, params):
  """Training loop."""
  ds = dataset.load_dataset(FLAGS.dataset_dir, 'train')
  ds = dataset.add_targets(ds, [params['field']],
                           add_history=params['history'])
  ds = dataset.split_and_preprocess(ds,
                                    noise_field=params['field'],
                                    noise_scale=params['noise'],
                                    noise_gamma=params['gamma'])
  inputs = tf.data.make_one_shot_iterator(ds).get_next()

  loss_op = model.loss(inputs)
  global_step = tf.train.create_global_step()
  lr = tf.train.exponential_decay(learning_rate=1e-4,
                                  global_step=global_step,
                                  decay_steps=int(5e6),
                                  decay_rate=0.1) + 1e-6
  optimizer = tf.train.AdamOptimizer(learning_rate=lr)
  train_op = optimizer.minimize(loss_op, global_step=global_step)

  # Warmup: for the first 1000 steps, only increment global_step and let the
  # normalisers accumulate statistics. After that, run the actual training op.
  train_op = tf.cond(tf.less(global_step, 1000),
                     lambda: tf.group(tf.assign_add(global_step, 1)),
                     lambda: tf.group(train_op))

  # --- Persist the training loss to a CSV file for offline analysis ---
  # (Written every 100 steps below inside the training loop.)
  os.makedirs(FLAGS.checkpoint_dir, exist_ok=True)
  loss_csv_path = os.path.join(FLAGS.checkpoint_dir, 'loss_history.csv')
  if not os.path.exists(loss_csv_path):
    with open(loss_csv_path, 'w') as f:
      f.write("step,loss\n")

  with tf.train.MonitoredTrainingSession(
      hooks=[tf.train.StopAtStepHook(last_step=FLAGS.num_training_steps)],
      checkpoint_dir=FLAGS.checkpoint_dir,
      save_checkpoint_secs=600) as sess:

    while not sess.should_stop():
      _, step, loss = sess.run([train_op, global_step, loss_op])
      if step % 100 == 0:
        logging.info('Step %d: Loss %g', step, loss)
        with open(loss_csv_path, 'a') as f:
          f.write(f"{step},{loss}\n")
    logging.info('Training complete.')


def evaluator(model, params):
  """Run rollout evaluation over `num_rollouts` trajectories."""
  ds = dataset.load_dataset(FLAGS.dataset_dir, FLAGS.rollout_split)
  ds = dataset.add_targets(ds, [params['field']],
                           add_history=params['history'])
  inputs = tf.data.make_one_shot_iterator(ds).get_next()
  scalar_op, traj_ops = params['evaluator'].evaluate(model, inputs)
  tf.train.create_global_step()

  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=FLAGS.checkpoint_dir,
      save_checkpoint_secs=None,
      save_checkpoint_steps=None) as sess:
    trajectories = []
    scalars = []
    for traj_idx in range(FLAGS.num_rollouts):
      logging.info('Rollout trajectory %d', traj_idx)
      scalar_data, traj_data = sess.run([scalar_op, traj_ops])
      trajectories.append(traj_data)
      scalars.append(scalar_data)
    for key in scalars[0]:
      logging.info('%s: %g', key, np.mean([x[key] for x in scalars]))
    with open(FLAGS.rollout_path, 'wb') as fp:
      pickle.dump(trajectories, fp)


def main(argv):
  del argv
  tf.enable_resource_variables()
  tf.disable_eager_execution()
  params = PARAMETERS[FLAGS.model]

  # Model configuration:
  # - latent_size=128 (split into 112 angular bins + 16 SELF dimensions)
  # - message_passing_steps=5 (M). Change here to experiment with other
  #   values of M. In the proposed architecture, the total number of
  #   parameters is independent of M.
  learned_model = core_model.EncodeProcessDecode(
      output_size=params['size'],
      latent_size=128,
      num_layers=2,
      message_passing_steps=5)
  model = params['model'].Model(learned_model)
  if FLAGS.mode == 'train':
    learner(model, params)
  elif FLAGS.mode == 'eval':
    evaluator(model, params)


if __name__ == '__main__':
  app.run(main)