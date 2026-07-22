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
# ============================================================================
"""Commonly used data structures and functions.

Defines the NodeType enum used across CFD and cloth models, and the
triangles_to_edges utility that converts a triangular mesh into a set of
directed edges suitable for GNN processing.
"""

import enum
import tensorflow.compat.v1 as tf


class NodeType(enum.IntEnum):
  """Node type labels used in the input datasets.

  For CylinderFlow, only NORMAL, OBSTACLE, INFLOW, OUTFLOW and
  WALL_BOUNDARY are used. AIRFOIL and HANDLE are kept for compatibility
  with other MeshGraphNets datasets.
  """
  NORMAL = 0
  OBSTACLE = 1
  AIRFOIL = 2
  HANDLE = 3
  INFLOW = 4
  OUTFLOW = 5
  WALL_BOUNDARY = 6
  SIZE = 9


def triangles_to_edges(faces):
  """Compute the set of directed edges of a triangular mesh.

  Each triangle (A, B, C) produces three undirected edges (A-B, B-C, C-A).
  Duplicate edges (shared between adjacent triangles) are removed by
  packing each pair of node ids into a single int64 and applying tf.unique.
  Finally, the graph is made bidirectional so that messages can flow in
  both directions along each edge.
  """
  # Split each triangle into its three edges
  edges = tf.concat([faces[:, 0:2],
                     faces[:, 1:3],
                     tf.stack([faces[:, 2], faces[:, 0]], axis=1)], axis=0)

  # Canonical ordering: (min, max) so that (A, B) and (B, A) collide
  receivers = tf.reduce_min(edges, axis=1)
  senders = tf.reduce_max(edges, axis=1)

  # Pack each (sender, receiver) pair into a single int64 for fast unique()
  packed_edges = tf.bitcast(tf.stack([senders, receivers], axis=1), tf.int64)
  unique_edges = tf.bitcast(tf.unique(packed_edges)[0], tf.int32)
  senders, receivers = tf.unstack(unique_edges, axis=1)

  # Return both directions of each edge for bidirectional message passing
  return (tf.concat([senders, receivers], axis=0),
          tf.concat([receivers, senders], axis=0))