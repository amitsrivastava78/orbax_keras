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

"""Tests for Keras Save Decision Policies with all backends."""

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
from orbax.checkpoint._src.checkpoint_managers import save_decision_policy
from orbax.checkpoint._src.handlers import keras_checkpoint_handler

try:
  import keras
  from keras import layers
  KERAS_AVAILABLE = True
except ImportError:
  keras = None
  KERAS_AVAILABLE = False


@unittest.skipIf(not KERAS_AVAILABLE, "Keras not available")
class KerasSaveDecisionPoliciesTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("jax", "jax"),
      ("tensorflow", "tensorflow"),
      ("torch", "torch"),
  )
  def test_fixed_interval_policy(self, backend):
    """Test FixedIntervalPolicy with Keras models across all backends."""
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
            save_decision_policy=save_decision_policy.FixedIntervalPolicy(interval=3),
            enable_async_checkpointing=False
        )

        manager = checkpoint_manager.CheckpointManager(
            directory,
            item_names=['model'],
            item_handlers={'model': keras_checkpoint_handler.KerasCheckpointHandler()},
            options=options
        )

        model = self._create_test_model()

        # Save at steps 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        # Should save at steps: 0, 3, 6, 9 (every 3 steps)
        expected_saved_steps = [0, 3, 6, 9]

        for step in range(10):
          # Simulate training
          current_weights = model.get_weights()
          new_weights = [w + np.random.normal(0, 0.01, w.shape) for w in current_weights]
          model.set_weights(new_weights)

          # Save checkpoint
          saved = manager.save(
              step,
              args=args_lib.Composite(
                  model=keras_checkpoint_handler.KerasSaveArgs(model)
              )
          )

          # Check if save was expected
          if step in expected_saved_steps:
            self.assertTrue(saved, f"Expected to save at step {step}")
          else:
            self.assertFalse(saved, f"Did not expect to save at step {step}")

        # Verify final checkpoints
        all_steps = manager.all_steps()
        self.assertEqual(sorted(all_steps), expected_saved_steps)

    finally:
      # Restore original backend
      if original_backend:
        os.environ['KERAS_BACKEND'] = original_backend

  @parameterized.named_parameters(
      ("jax", "jax"),
      ("tensorflow", "tensorflow"),
      ("torch", "torch"),
  )
  def test_specific_steps_policy(self, backend):
    """Test SpecificStepsPolicy with Keras models across all backends."""
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

        # Save at specific steps: 2, 5, 8
        options = checkpoint_manager.CheckpointManagerOptions(
            save_decision_policy=save_decision_policy.SpecificStepsPolicy(steps=[2, 5, 8]),
            enable_async_checkpointing=False
        )

        manager = checkpoint_manager.CheckpointManager(
            directory,
            item_names=['model'],
            item_handlers={'model': keras_checkpoint_handler.KerasCheckpointHandler()},
            options=options
        )

        model = self._create_test_model()
        expected_saved_steps = [2, 5, 8]

        for step in range(10):
          # Simulate training
          current_weights = model.get_weights()
          new_weights = [w + np.random.normal(0, 0.01, w.shape) for w in current_weights]
          model.set_weights(new_weights)

          # Save checkpoint
          saved = manager.save(
              step,
              args=args_lib.Composite(
                  model=keras_checkpoint_handler.KerasSaveArgs(model)
              )
          )

          # Check if save was expected
          if step in expected_saved_steps:
            self.assertTrue(saved, f"Expected to save at step {step}")
          else:
            self.assertFalse(saved, f"Did not expect to save at step {step}")

        # Verify final checkpoints
        all_steps = manager.all_steps()
        self.assertEqual(sorted(all_steps), expected_saved_steps)

    finally:
      # Restore original backend
      if original_backend:
        os.environ['KERAS_BACKEND'] = original_backend

  @parameterized.named_parameters(
      ("jax", "jax"),
      ("tensorflow", "tensorflow"),
      ("torch", "torch"),
  )
  def test_initial_save_policy(self, backend):
    """Test InitialSavePolicy with Keras models across all backends."""
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
            save_decision_policy=save_decision_policy.InitialSavePolicy(),
            enable_async_checkpointing=False
        )

        manager = checkpoint_manager.CheckpointManager(
            directory,
            item_names=['model'],
            item_handlers={'model': keras_checkpoint_handler.KerasCheckpointHandler()},
            options=options
        )

        model = self._create_test_model()

        # First save should succeed (no previous checkpoints)
        saved = manager.save(
            0,
            args=args_lib.Composite(
                model=keras_checkpoint_handler.KerasSaveArgs(model)
            )
        )
        self.assertTrue(saved, "Expected to save initial checkpoint")

        # Subsequent saves should not happen (checkpoints already exist)
        for step in range(1, 5):
          # Simulate training
          current_weights = model.get_weights()
          new_weights = [w + np.random.normal(0, 0.01, w.shape) for w in current_weights]
          model.set_weights(new_weights)

          saved = manager.save(
              step,
              args=args_lib.Composite(
                  model=keras_checkpoint_handler.KerasSaveArgs(model)
              )
          )
          self.assertFalse(saved, f"Did not expect to save at step {step} (checkpoints already exist)")

        # Verify only initial checkpoint exists
        all_steps = manager.all_steps()
        self.assertEqual(all_steps, [0])

    finally:
      # Restore original backend
      if original_backend:
        os.environ['KERAS_BACKEND'] = original_backend

  @parameterized.named_parameters(
      ("jax", "jax"),
      ("tensorflow", "tensorflow"),
      ("torch", "torch"),
  )
  def test_any_save_policy(self, backend):
    """Test AnySavePolicy combining multiple policies with Keras models."""
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

        # Combine InitialSavePolicy and FixedIntervalPolicy
        combined_policy = save_decision_policy.AnySavePolicy([
            save_decision_policy.InitialSavePolicy(),  # Save first checkpoint
            save_decision_policy.FixedIntervalPolicy(interval=4)  # Save every 4 steps
        ])

        options = checkpoint_manager.CheckpointManagerOptions(
            save_decision_policy=combined_policy,
            enable_async_checkpointing=False
        )

        manager = checkpoint_manager.CheckpointManager(
            directory,
            item_names=['model'],
            item_handlers={'model': keras_checkpoint_handler.KerasCheckpointHandler()},
            options=options
        )

        model = self._create_test_model()

        # For steps 0-11:
        # - Step 0: InitialSavePolicy saves (first checkpoint)
        # - Steps 4, 8: FixedIntervalPolicy saves (every 4 steps)
        # - Other steps: No save
        expected_saved_steps = [0, 4, 8]

        for step in range(12):
          # Simulate training
          current_weights = model.get_weights()
          new_weights = [w + np.random.normal(0, 0.01, w.shape) for w in current_weights]
          model.set_weights(new_weights)

          # Save checkpoint
          saved = manager.save(
              step,
              args=args_lib.Composite(
                  model=keras_checkpoint_handler.KerasSaveArgs(model)
              )
          )

          # Check if save was expected
          if step in expected_saved_steps:
            self.assertTrue(saved, f"Expected to save at step {step}")
          else:
            self.assertFalse(saved, f"Did not expect to save at step {step}")

        # Verify final checkpoints
        all_steps = manager.all_steps()
        self.assertEqual(sorted(all_steps), expected_saved_steps)

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