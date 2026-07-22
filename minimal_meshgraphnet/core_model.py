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
#   - Reinterpretation of the 128-dim latent space as 112 angular bins + 16
#     SELF dimensions.
#   - Shared MLP_edge across all M message-passing steps.
#   - Deterministic angular scatter (replaces the aggregation MLP_node).
#   - Reflexive edges (i -> i) to populate the SELF dimensions.
#   - No residual connection in the node update.
# Bachelor's Thesis - Universidad de Zaragoza (EINA) - 2026
# Reference: T. Pfaff et al., "Learning Mesh-Based Simulation with Graph
# Networks", ICLR 2021. arXiv:2010.03409
# ============================================================================
"""Core learned graph net model with reinterpretation of the latent space.

Implements a minimal MeshGraphNet variant that reinterprets the 128-dim
latent space as a discretised angular distribution around each node
(112 directional bins + 16 SELF dimensions).
"""

import collections
import sonnet as snt
import tensorflow.compat.v1 as tf

EdgeSet = collections.namedtuple(
    'EdgeSet',
    ['name', 'features', 'senders', 'receivers',
     'geometry', 'sender_node_type',
     'bin_start', 'bin_width'])
MultiGraph = collections.namedtuple('Graph', ['node_features', 'edge_sets'])

# Latent space split: 112 angular bins + 16 SELF dimensions
LATENT_TOTAL = 128
ANGULAR_DIMS = 112
SELF_DIMS    = 16


def precompute_scatter_indices(edge_set):
  """Precompute indices for the angular scatter operation.

  Given the bin_start and bin_width tensors for each edge (produced by the
  Sinkhorn-Knopp preprocessing in the dataset), returns the tensor indices
  needed for the scatter_nd operation in the node update step.
  """
  n_edges = tf.shape(edge_set.bin_start)[0]
  bin_width = edge_set.bin_width
  bin_start = edge_set.bin_start
  receivers = edge_set.receivers

  offsets_ragged = tf.ragged.range(bin_width)
  offsets_flat = offsets_ragged.flat_values

  edge_idx_v = tf.repeat(tf.range(n_edges), bin_width)
  recv_idx_v = tf.repeat(receivers, bin_width)
  bin_start_per_bin = tf.repeat(bin_start, bin_width)
  bin_idx_v = tf.mod(bin_start_per_bin + offsets_flat, ANGULAR_DIMS)

  return {
      'bin_idx':  bin_idx_v,
      'recv_idx': recv_idx_v,
      'edge_idx': edge_idx_v,
  }


class GraphNetBlock(snt.AbstractModule):
  """GNN block with reflexive edges and no residual connection.

  Design decisions:

  1. No MLP_self. The self-information of each node is fed through the same
     shared MLP_edge as regular edges.
  2. Each node has an added reflexive edge (i -> i) that passes through the
     shared MLP_edge, with a null geometry (0, 0, 0) and the node's own type.
  3. The reflexive message is scattered into latent dimensions [112, 128).
  4. No residual connection in the node update: the new latent replaces the
     previous one entirely.
  5. The angular scatter operation replaces the aggregation MLP_node of the
     original MeshGraphNets architecture.
  """

  def __init__(self, shared_edge_mlp, scatter_indices,
               self_geometry, self_node_type,
               name='GraphNetBlock'):
    super(GraphNetBlock, self).__init__(name=name)
    self._shared_edge_mlp = shared_edge_mlp
    self._scatter_indices = scatter_indices
    # Reflexive edge features (already normalised by the encoder):
    #   self_geometry:    [N, 3]   - always (0, 0, 0) normalised
    #   self_node_type:   [N, T]   - one-hot of the node's own type
    self._self_geometry  = self_geometry
    self._self_node_type = self_node_type

  def _update_edge_features(self, node_features, edge_set):
    """Step 1.1: compute messages for real edges (i, j) with i != j."""
    sender_features = tf.gather(node_features, edge_set.senders)
    features = tf.concat(
        [sender_features, edge_set.geometry, edge_set.sender_node_type],
        axis=-1)
    return self._shared_edge_mlp(features)

  def _compute_reflexive_messages(self, node_features):
    """Step 1.2: compute reflexive messages (i -> i) for every node.

    Each node sends a message to itself through the same shared MLP_edge as
    the real edges, with a null geometry and its own node type. The result
    populates the SELF portion of the latent space.
    """
    features = tf.concat(
        [node_features, self._self_geometry, self._self_node_type],
        axis=-1)
    return self._shared_edge_mlp(features)   # [N, 128]

  def _update_node_features(self, node_features, edge_set, reflexive_messages):
    """Step 2: angular scatter and insertion of e_ii into dims [112, 128).

    No residual connection is applied: the node's new latent state fully
    replaces the previous one.
    """
    num_nodes = tf.shape(node_features)[0]
    idx = self._scatter_indices

    # === Angular scatter over dims [0, 112) ===
    origen = tf.stack([idx['edge_idx'], idx['bin_idx']], axis=1)
    valores = tf.gather_nd(edge_set.features, origen)
    destino = tf.stack([idx['recv_idx'], idx['bin_idx']], axis=1)
    latente_angular = tf.scatter_nd(
        destino, valores,
        shape=tf.stack([num_nodes, ANGULAR_DIMS]))                  # [N, 112]

    # === Insert e_ii into dims [112, 128) ===
    # reflexive_messages has 128 dims; we only keep the last 16.
    self_part = reflexive_messages[:, ANGULAR_DIMS:]                # [N, 16]

    # === Compose the new latent state ===
    latente_nuevo = tf.concat([latente_angular, self_part], axis=1) # [N, 128]
    return latente_nuevo

  def _build(self, graph):
    # Step 1.1: real edges
    new_edge_sets = []
    for edge_set in graph.edge_sets:
      updated_features = self._update_edge_features(
          graph.node_features, edge_set)
      new_edge_sets.append(edge_set._replace(
          features=updated_features,
          geometry=edge_set.geometry,
          sender_node_type=edge_set.sender_node_type,
          bin_start=edge_set.bin_start,
          bin_width=edge_set.bin_width))

    # Step 1.2: reflexive edges
    reflexive_messages = self._compute_reflexive_messages(graph.node_features)

    # Step 2: node update (no residual)
    new_node_features = self._update_node_features(
        graph.node_features, new_edge_sets[0], reflexive_messages)

    # The residual connection h_i^(m) + delta was removed in this variant.

    # Residual connection on edges (kept, as in the original architecture)
    new_edge_sets = [
        es._replace(
            features=es.features + old_es.features,
            geometry=es.geometry,
            sender_node_type=es.sender_node_type,
            bin_start=es.bin_start,
            bin_width=es.bin_width)
        for es, old_es in zip(new_edge_sets, graph.edge_sets)]

    return MultiGraph(new_node_features, new_edge_sets)


class EncodeProcessDecode(snt.AbstractModule):
  """Encode-Process-Decode architecture with angular latent reinterpretation."""

  def __init__(self, output_size, latent_size, num_layers, message_passing_steps,
               num_layers_edge=None,
               name='EncodeProcessDecode'):
    super(EncodeProcessDecode, self).__init__(name=name)
    self._latent_size = latent_size
    self._output_size = output_size
    self._num_layers = num_layers
    self._message_passing_steps = message_passing_steps
    self._num_layers_edge = (num_layers_edge if num_layers_edge is not None
                             else num_layers)

  def _make_mlp(self, output_size, layer_norm=True, num_layers=None):
    """Build a multi-layer perceptron with optional layer normalisation."""
    n = num_layers if num_layers is not None else self._num_layers
    widths = [self._latent_size] * n + [output_size]
    network = snt.nets.MLP(widths, activate_final=False)
    if layer_norm:
      network = snt.Sequential([network, snt.LayerNorm()])
    return network

  def _encoder(self, graph):
    """Encode node and edge raw features into the 128-dim latent space."""
    with tf.variable_scope('encoder'):
      node_latents = self._make_mlp(self._latent_size)(graph.node_features)
      new_edges_sets = []
      for edge_set in graph.edge_sets:
        latent = self._make_mlp(self._latent_size)(edge_set.features)
        new_edges_sets.append(edge_set._replace(
            features=latent,
            geometry=edge_set.geometry,
            sender_node_type=edge_set.sender_node_type,
            bin_start=edge_set.bin_start,
            bin_width=edge_set.bin_width))
    return MultiGraph(node_latents, new_edges_sets)

  def _decoder(self, graph):
    """Decode the final node latent state into the output prediction."""
    with tf.variable_scope('decoder'):
      decoder = self._make_mlp(self._output_size, layer_norm=False)
      return decoder(graph.node_features)

  def _build(self, graph, self_geometry=None, self_node_type=None):
    """Run the full Encode-Process-Decode forward pass.

    Args:
      graph: MultiGraph containing the real edges of the mesh.
      self_geometry: [N, 3] geometric features for the reflexive edges,
                     typically (0, 0, 0) after normalisation. If None,
                     defaults to zeros.
      self_node_type: [N, T] one-hot encoding of each node's own type,
                     used as the sender type for the reflexive edge.
    """
    latent_graph = self._encoder(graph)

    # Shared MLP_edge across all message-passing steps
    with tf.variable_scope('shared_edge_mlp'):
      shared_edge_mlp = self._make_mlp(
          self._latent_size,
          num_layers=self._num_layers_edge)

    # Precompute scatter indices for real edges
    assert len(latent_graph.edge_sets) == 1
    scatter_indices = precompute_scatter_indices(latent_graph.edge_sets[0])

    # Default self_geometry to zeros if not provided
    num_nodes = tf.shape(latent_graph.node_features)[0]
    if self_geometry is None:
      # 3 dims: dx=0, dy=0, norm=0
      self_geometry = tf.zeros([num_nodes, 3], dtype=tf.float32)
    if self_node_type is None:
      raise ValueError("self_node_type must be provided (one-hot).")

    # Processor: M message-passing steps sharing the same MLP_edge
    for _ in range(self._message_passing_steps):
      latent_graph = GraphNetBlock(
          shared_edge_mlp, scatter_indices,
          self_geometry, self_node_type)(latent_graph)

    return self._decoder(latent_graph)