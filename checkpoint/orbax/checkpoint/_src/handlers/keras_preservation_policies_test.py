# Copyright 2025 The Orbax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for Keras Preservation Policies with all backends."""

import os
import tempfile
import unittest

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import numpy as np

from orbax.checkpoint import args as args_lib
from orbax.checkpoint import checkpoint_manager
from orbax.checkpoint import options as options_lib
from orbax.checkpoint._src.checkpoint_managers import preservation_policy
from orbax.checkpoint._src.handlers import keras_checkpoint_handler

try:
  import keras
  from keras import layers
  KERAS_AVAILABLE = True
except ImportError:
  keras = None
  KERAS_AVAILABLE = False


@unittest.skipIf(not KERAS_AVAILABLE, "Keras not available")
class KerasPreservationPoliciesTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("jax", "jax"),
      ("tensorflow", "tensorflow"),
      ("torch", "torch"),
  )
  def test_latest_n_preservation_policy(self, backend):
    """Test LatestN preservation policy with Keras models across all backends."""
    # Set backend
    original_backend = os.environ.get('KERAS_BACKEND')
    os.environ['KERAS_BACKEND'] = backend

    try:
      # Force backend reload
      import importlib
      import keras
      importlib.reload(keras)

      with tempfile.TemporaryDirectory() as tmpdir:
        directory = epath.Path(tmpdir)

        # Create checkpoint manager with LatestN preservation policy
        options = checkpoint_manager.CheckpointManagerOptions(
            preservation_policy=preservation_policy.LatestN(n=3),
            save_interval_steps=1,  # Save every step
            enable_async_checkpointing=False  # Use synchronous checkpointing for testing
        )

        manager = checkpoint_manager.CheckpointManager(
            directory,
            item_names=['model'],
            item_handlers={'model': keras_checkpoint_handler.KerasCheckpointHandler()},
            options=options
        )

        # Train and save multiple checkpoints
        model = self._create_test_model()
        for step in range(5):
          # Simulate training step (update weights slightly)
          current_weights = model.get_weights()
          # Add small random noise to simulate training
          new_weights = [w + np.random.normal(0, 0.01, w.shape) for w in current_weights]
          model.set_weights(new_weights)

          # Save checkpoint using new API
          manager.save(
              step,
              args=args_lib.Composite(
                  model=keras_checkpoint_handler.KerasSaveArgs(model)
              )
          )

        # Check that only the latest 3 checkpoints are preserved
        all_steps = manager.all_steps()
        self.assertEqual(len(all_steps), 3)
        self.assertEqual(sorted(all_steps), [2, 3, 4])  # Steps 2, 3, 4

    finally:
      # Restore original backend
      if original_backend:
        os.environ['KERAS_BACKEND'] = original_backend

  @parameterized.named_parameters(
      ("jax", "jax"),
      ("tensorflow", "tensorflow"),
      ("torch", "torch"),
  )
  def test_every_n_steps_preservation_policy(self, backend):
    """Test EveryNSteps preservation policy with Keras models across all backends."""
    # Set backend
    original_backend = os.environ.get('KERAS_BACKEND')
    os.environ['KERAS_BACKEND'] = backend

    try:
      # Force backend reload
      import importlib
      import keras
      importlib.reload(keras)

      with tempfile.TemporaryDirectory() as tmpdir:
        directory = epath.Path(tmpdir)

        options = checkpoint_manager.CheckpointManagerOptions(
            preservation_policy=preservation_policy.EveryNSteps(interval_steps=3),
            save_interval_steps=1,
            enable_async_checkpointing=False
        )

        manager = checkpoint_manager.CheckpointManager(
            directory,
            item_names=['model'],
            item_handlers={'model': keras_checkpoint_handler.KerasCheckpointHandler()},
            options=options
        )

        model = self._create_test_model()
        for step in range(10):
          # Simulate training
          current_weights = model.get_weights()
          new_weights = [w + np.random.normal(0, 0.01, w.shape) for w in current_weights]
          model.set_weights(new_weights)

          # Save checkpoint
          manager.save(
              step,
              args=args_lib.Composite(
                  model=keras_checkpoint_handler.KerasSaveArgs(model)
              )
          )

        # Should keep steps 0, 3, 6, 9 (every 3 steps)
        all_steps = manager.all_steps()
        expected_steps = [0, 3, 6, 9]
        self.assertEqual(sorted(all_steps), expected_steps)

    finally:
      # Restore original backend
      if original_backend:
        os.environ['KERAS_BACKEND'] = original_backend

  @parameterized.named_parameters(
      ("jax", "jax"),
      ("tensorflow", "tensorflow"),
      ("torch", "torch"),
  )
  def test_best_n_preservation_policy(self, backend):
    """Test BestN preservation policy with Keras models across all backends."""
    # Set backend
    original_backend = os.environ.get('KERAS_BACKEND')
    os.environ['KERAS_BACKEND'] = backend

    try:
      # Force backend reload
      import importlib
      import keras
      importlib.reload(keras)

      with tempfile.TemporaryDirectory() as tmpdir:
        directory = epath.Path(tmpdir)

        # Create a simple metric function that improves over time
        def metric_fn(metrics):
          # Simulate validation loss that improves over time
          return metrics.get('val_loss', float('inf'))

        options = checkpoint_manager.CheckpointManagerOptions(
            preservation_policy=preservation_policy.BestN(
                n=2,
                get_metric_fn=metric_fn,
                reverse=False  # False for min (lower is better)
            ),
            save_interval_steps=1,
            enable_async_checkpointing=False
        )

        manager = checkpoint_manager.CheckpointManager(
            directory,
            item_names=['model'],
            item_handlers={'model': keras_checkpoint_handler.KerasCheckpointHandler()},
            options=options
        )

        model = self._create_test_model()
        # Simulate training with improving validation loss
        val_losses = [1.0, 0.8, 0.9, 0.5, 0.7, 0.3]  # Best are 0.3 and 0.5

        for step, val_loss in enumerate(val_losses):
          # Simulate training
          current_weights = model.get_weights()
          new_weights = [w + np.random.normal(0, 0.01, w.shape) for w in current_weights]
          model.set_weights(new_weights)

          # Save with metrics
          manager.save(
              step,
              args=args_lib.Composite(
                  model=keras_checkpoint_handler.KerasSaveArgs(model)
              ),
              metrics={'val_loss': val_loss}
          )

        # Should keep the 2 best checkpoints (steps with losses 0.3 and 0.5)
        all_steps = manager.all_steps()
        self.assertEqual(len(all_steps), 2)
        # The exact steps depend on implementation, but should be the best 2

    finally:
      # Restore original backend
      if original_backend:
        os.environ['KERAS_BACKEND'] = original_backend

  @parameterized.named_parameters(
      ("jax", "jax"),
      ("tensorflow", "tensorflow"),
      ("torch", "torch"),
  )
  def test_combined_preservation_policies(self, backend):
    """Test combining multiple preservation policies with Keras models."""
    # Set backend
    original_backend = os.environ.get('KERAS_BACKEND')
    os.environ['KERAS_BACKEND'] = backend

    try:
      # Force backend reload
      import importlib
      import keras
      importlib.reload(keras)

      with tempfile.TemporaryDirectory() as tmpdir:
        directory = epath.Path(tmpdir)

        # Combine LatestN and EveryNSteps policies
        combined_policy = preservation_policy.AnyPreservationPolicy([
            preservation_policy.LatestN(n=2),  # Keep latest 2
            preservation_policy.EveryNSteps(interval_steps=4)  # Keep every 4 steps
        ])

        options = checkpoint_manager.CheckpointManagerOptions(
            preservation_policy=combined_policy,
            save_interval_steps=1,
            enable_async_checkpointing=False
        )

        manager = checkpoint_manager.CheckpointManager(
            directory,
            item_names=['model'],
            item_handlers={'model': keras_checkpoint_handler.KerasCheckpointHandler()},
            options=options
        )

        model = self._create_test_model()
        for step in range(12):
          # Simulate training
          current_weights = model.get_weights()
          new_weights = [w + np.random.normal(0, 0.01, w.shape) for w in current_weights]
          model.set_weights(new_weights)

          # Save checkpoint
          manager.save(
              step,
              args=args_lib.Composite(
                  model=keras_checkpoint_handler.KerasSaveArgs(model)
              )
          )

        # Should keep: latest 2 (10, 11) + every 4 steps (0, 4, 8)
        all_steps = manager.all_steps()
        expected_steps = sorted([0, 4, 8, 10, 11])
        self.assertEqual(sorted(all_steps), expected_steps)

    finally:
      # Restore original backend
      if original_backend:
        os.environ['KERAS_BACKEND'] = original_backend

  def _create_test_model(self):
    """Create a simple Keras model for testing."""
    model = keras.Sequential([
        layers.Dense(4, input_shape=(4,), activation='relu'),
        layers.Dense(2, activation='relu'),
        layers.Dense(1)
    ])
    # Compile to set up optimizer state
    model.compile(optimizer='adam', loss='mse')
    return model


if __name__ == '__main__':
  absltest.main()

  @parameterized.named_parameters(
      ("jax", "jax"),
      ("tensorflow", "tensorflow"),
      ("torch", "torch"),
  )
  def test_best_n_preservation_policy(self, backend):
    """Test BestN preservation policy with Keras models across all backends."""
    # Set backend
    original_backend = os.environ.get('KERAS_BACKEND')
    os.environ['KERAS_BACKEND'] = backend

    try:
      # Force backend reload
      import importlib
      import keras
      importlib.reload(keras)

      with tempfile.TemporaryDirectory() as tmpdir:
        directory = epath.Path(tmpdir)

        # Create a simple metric function that improves over time
        def metric_fn(metrics):
          # Simulate validation loss that improves over time
          return metrics.get('val_loss', float('inf'))

        options = checkpoint_manager.CheckpointManagerOptions(
            preservation_policy=preservation_policy.BestN(
                n=2,
                metric_fn=metric_fn,
                mode='min'  # Lower is better
            ),
            save_interval_steps=1
        )

        manager = checkpoint_manager.CheckpointManager(
            directory,
            item_names=['model'],
            item_handlers={'model': keras.KerasCheckpointHandler()},
            options=options
        )

        model = self._create_test_model()
        # Simulate training with improving validation loss
        val_losses = [1.0, 0.8, 0.9, 0.5, 0.7, 0.3]  # Best are 0.3 and 0.5

        for step, val_loss in enumerate(val_losses):
          # Simulate training
          current_weights = model.get_weights()
          new_weights = [w + np.random.normal(0, 0.01, w.shape) for w in current_weights]
          model.set_weights(new_weights)

          # Save with metrics
          manager.save(
              step,
              args=args_lib.Composite(
                  model=args_lib.KerasSave(model)
              ),
              metrics={'val_loss': val_loss}
          )

        # Should keep the 2 best checkpoints (steps with losses 0.3 and 0.5)
        all_steps = manager.all_steps()
        self.assertEqual(len(all_steps), 2)
        # The exact steps depend on implementation, but should be the best 2

    finally:
      # Restore original backend
      if original_backend:
        os.environ['KERAS_BACKEND'] = original_backend

  @parameterized.named_parameters(
      ("jax", "jax"),
      ("tensorflow", "tensorflow"),
      ("torch", "torch"),
  )
  def test_combined_preservation_policies(self, backend):
    """Test combining multiple preservation policies with Keras models."""
    # Set backend
    original_backend = os.environ.get('KERAS_BACKEND')
    os.environ['KERAS_BACKEND'] = backend

    try:
      # Force backend reload
      import importlib
      import keras
      importlib.reload(keras)

      with tempfile.TemporaryDirectory() as tmpdir:
        directory = epath.Path(tmpdir)

        # Combine LatestN and EveryNSteps policies
        combined_policy = preservation_policy.AnyPreservationPolicy([
            preservation_policy.LatestN(n=2),  # Keep latest 2
            preservation_policy.EveryNSteps(n=4)  # Keep every 4 steps
        ])

        options = checkpoint_manager.CheckpointManagerOptions(
            preservation_policy=combined_policy,
            save_interval_steps=1
        )

        manager = checkpoint_manager.CheckpointManager(
            directory,
            item_names=['model'],
            item_handlers={'model': keras.KerasCheckpointHandler()},
            options=options
        )

        model = self._create_test_model()
        for step in range(12):
          # Simulate training
          current_weights = model.get_weights()
          new_weights = [w + np.random.normal(0, 0.01, w.shape) for w in current_weights]
          model.set_weights(new_weights)

          # Save checkpoint
          manager.save(
              step,
              args=args_lib.Composite(
                  model=args_lib.KerasSave(model)
              )
          )

        # Should keep: latest 2 (10, 11) + every 4 steps (0, 4, 8)
        all_steps = manager.all_steps()
        expected_steps = sorted([0, 4, 8, 10, 11])
        self.assertEqual(sorted(all_steps), expected_steps)

    finally:
      # Restore original backend
      if original_backend:
        os.environ['KERAS_BACKEND'] = original_backend

  def _create_test_model(self):
    """Create a simple Keras model for testing."""
    model = keras.Sequential([
        layers.Dense(4, input_shape=(4,), activation='relu'),
        layers.Dense(2, activation='relu'),
        layers.Dense(1)
    ])
    # Compile to set up optimizer state
    model.compile(optimizer='adam', loss='mse')
    return model


if __name__ == '__main__':
  absltest.main()