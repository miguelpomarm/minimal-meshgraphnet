# pylint: disable=g-bad-file-header
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ============================================================================
# Modifications by Miguel Pomar Martinez (2026):
#   - Build synthetic reflexive edge features (i -> i) with zero geometry
#     and each node's own type.
#   - Concatenate real and reflexive edges before normalisation so that the
#     edge_normalizer statistics are learned over the joint distribution.
#   - Extract precomputed angular bin ranges (edge_bin_start, edge_bin_width)
#     from the input, which the Sinkhorn-Knopp preprocessing step produces
#     for each real edge.
#   - Pass self_geometry and self_node_type down to the learned model.
# Bachelor's Thesis - Universidad de Zaragoza (EINA) - 2026
# Reference: T. Pfaff et al., "Learning Mesh-Based Simulation with Graph
# Networks", ICLR 2021. arXiv:2010.03409
# ============================================================================
"""Model for CylinderFlow (2D incompressible flow around a cylinder).

Wraps the learned core model with the CFD-specific preprocessing:
input feature construction, edge extraction from mesh triangles,
normalisation, and loss/integration on the velocity field.
"""

import sonnet as snt
import tensorflow.compat.v1 as tf

from minimal_meshgraphnet import common
from minimal_meshgraphnet import core_model
from minimal_meshgraphnet import normalization


class Model(snt.AbstractModule):
  """CFD model wrapper around the learned Encode-Process-Decode network."""

  def __init__(self, learned_model, name='Model'):
    super(Model, self).__init__(name=name)
    with self._enter_variable_scope():
      self._learned_model = learned_model

      self._output_normalizer = normalization.Normalizer(
          size=2, name='output_normalizer')
      self._node_normalizer = normalization.Normalizer(
          size=2 + common.NodeType.SIZE, name='node_normalizer')
      self._edge_normalizer = normalization.Normalizer(
          size=3, name='edge_normalizer')

  def _build_graph(self, inputs, is_training):
    """Build the input MultiGraph from raw features.

    Returns:
      A tuple (graph, self_geometry, self_node_type) where:
        - graph:            MultiGraph with the mesh nodes and real edges.
        - self_geometry:    [N, 3] normalised zero-geometry for reflexive
                            edges (i -> i).
        - self_node_type:   [N, T] one-hot node type used as sender type
                            for reflexive edges.
    """
    # === Node features: velocity + one-hot node type ===
    node_type = tf.one_hot(inputs['node_type'][:, 0], common.NodeType.SIZE)
    node_features = tf.concat([inputs['velocity'], node_type], axis=-1)

    # === Real edges: extracted from the mesh triangles ===
    senders, receivers = common.triangles_to_edges(inputs['cells'])

    relative_mesh_pos = (tf.gather(inputs['mesh_pos'], senders) -
                         tf.gather(inputs['mesh_pos'], receivers))

    edge_features = tf.concat([
        relative_mesh_pos,
        tf.norm(relative_mesh_pos, axis=-1, keepdims=True)], axis=-1)

    # === Reflexive edges: synthetic (i -> i) edges with zero geometry ===
    num_nodes = tf.shape(node_type)[0]
    self_edge_features = tf.zeros(tf.stack([num_nodes, 3]),
                                  dtype=edge_features.dtype)

    # Normalise real and reflexive edges jointly so that the edge_normalizer
    # statistics account for the presence of reflexive edges.
    n_real_edges = tf.shape(edge_features)[0]
    all_edge_features = tf.concat([edge_features, self_edge_features], axis=0)
    all_normalized = self._edge_normalizer(all_edge_features, is_training)

    # Split back into real and reflexive after normalisation
    normalized_geometry      = all_normalized[:n_real_edges]
    normalized_self_geometry = all_normalized[n_real_edges:]

    # Sender node type (one-hot of the sending node's type)
    sender_node_type = tf.gather(node_type, senders)

    # Precomputed angular bin ranges (from the Sinkhorn-Knopp step in dataset)
    bin_start = tf.squeeze(inputs['edge_bin_start'], axis=-1)
    bin_width = tf.squeeze(inputs['edge_bin_width'], axis=-1)
    bin_start = tf.cast(bin_start, tf.int32)
    bin_width = tf.cast(bin_width, tf.int32)

    # EdgeSet of real edges
    mesh_edges = core_model.EdgeSet(
        name='mesh_edges',
        features=normalized_geometry,
        receivers=receivers,
        senders=senders,
        geometry=normalized_geometry,
        sender_node_type=sender_node_type,
        bin_start=bin_start,
        bin_width=bin_width)

    graph = core_model.MultiGraph(
        node_features=self._node_normalizer(node_features, is_training),
        edge_sets=[mesh_edges])

    # self_node_type for the reflexive edge = each node's own type
    self_node_type = node_type   # [N, T]

    return graph, normalized_self_geometry, self_node_type

  def _build(self, inputs):
    """Forward pass at inference time (used for rollout evaluation)."""
    graph, self_geom, self_type = self._build_graph(inputs, is_training=False)
    per_node_network_output = self._learned_model(
        graph, self_geometry=self_geom, self_node_type=self_type)
    return self._update(inputs, per_node_network_output)

  @snt.reuse_variables
  def loss(self, inputs):
    """L2 loss on the predicted velocity change.

    Applies the loss only over nodes of type NORMAL or OUTFLOW, i.e. it
    excludes boundary nodes (walls, inflow) whose velocity is fixed.
    """
    graph, self_geom, self_type = self._build_graph(inputs, is_training=True)
    network_output = self._learned_model(
        graph, self_geometry=self_geom, self_node_type=self_type)

    cur_velocity = inputs['velocity']
    target_velocity = inputs['target|velocity']
    target_velocity_change = target_velocity - cur_velocity
    target_normalized = self._output_normalizer(target_velocity_change)

    node_type = inputs['node_type'][:, 0]
    loss_mask = tf.logical_or(tf.equal(node_type, common.NodeType.NORMAL),
                              tf.equal(node_type, common.NodeType.OUTFLOW))

    error = tf.reduce_sum((target_normalized - network_output) ** 2, axis=1)
    loss = tf.reduce_mean(error[loss_mask])
    return loss

  def _update(self, inputs, per_node_network_output):
    """Integrate the predicted velocity change into the next state."""
    velocity_update = self._output_normalizer.inverse(per_node_network_output)
    cur_velocity = inputs['velocity']
    return cur_velocity + velocity_update