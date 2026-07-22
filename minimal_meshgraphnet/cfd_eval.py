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
"""Functions to build evaluation metrics for CFD data.

Provides the autoregressive rollout and per-horizon MSE metrics used to
evaluate the trained model on the CylinderFlow test set.
"""

import tensorflow.compat.v1 as tf

from minimal_meshgraphnet.common import NodeType


def _rollout(model, initial_state, num_steps):
  """Roll out a model trajectory autoregressively.

  Starting from the initial state, at each step the model predicts the
  next velocity field. Boundary nodes (INFLOW, WALL_BOUNDARY, OBSTACLE)
  are held fixed and only NORMAL and OUTFLOW nodes are updated.
  """
  node_type = initial_state['node_type'][:, 0]
  mask = tf.logical_or(tf.equal(node_type, NodeType.NORMAL),
                       tf.equal(node_type, NodeType.OUTFLOW))

  def step_fn(step, velocity, trajectory):
    prediction = model({**initial_state,
                        'velocity': velocity})
    # Do not update boundary nodes
    next_velocity = tf.where(mask, prediction, velocity)
    trajectory = trajectory.write(step, velocity)
    return step+1, next_velocity, trajectory

  _, _, output = tf.while_loop(
      cond=lambda step, cur, traj: tf.less(step, num_steps),
      body=step_fn,
      loop_vars=(0, initial_state['velocity'],
                 tf.TensorArray(tf.float32, num_steps)),
      parallel_iterations=1)
  return output.stack()


def evaluate(model, inputs):
  """Perform a model rollout and compute evaluation statistics.

  Returns:
    A tuple (scalars, traj_ops) where:
      - scalars: dict mapping 'mse_{horizon}_steps' to the mean squared
                 error between predicted and ground-truth velocity fields
                 over the first `horizon` steps.
      - traj_ops: dict with the mesh geometry, the ground-truth velocity
                  and the predicted velocity of the full trajectory. This
                  is what gets pickled by the evaluator for later analysis.
  """
  initial_state = {k: v[0] for k, v in inputs.items()}
  num_steps = inputs['cells'].shape[0]
  prediction = _rollout(model, initial_state, num_steps)

  error = tf.reduce_mean((prediction - inputs['velocity'])**2, axis=-1)
  scalars = {'mse_%d_steps' % horizon: tf.reduce_mean(error[1:horizon+1])
             for horizon in [1, 10, 20, 50, 100, 200]}
  traj_ops = {
      'faces': inputs['cells'],
      'mesh_pos': inputs['mesh_pos'],
      'gt_velocity': inputs['velocity'],
      'pred_velocity': prediction
  }
  return scalars, traj_ops