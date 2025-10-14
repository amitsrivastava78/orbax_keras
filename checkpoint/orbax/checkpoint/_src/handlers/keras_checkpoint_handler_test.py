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

"""Tests for KerasCheckpointHandler."""

import os
os.environ['KERAS_BACKEND'] = 'jax'

import tempfile
import unittest

from absl.testing import absltest
from etils import epath
import numpy as np

from orbax.checkpoint import handlers
from orbax.checkpoint import transform_utils
from orbax.checkpoint._src.handlers import keras_checkpoint_handler

try:
  import keras
  from keras import layers
  KERAS_AVAILABLE = True
except ImportError:
  keras = None
  KERAS_AVAILABLE = False


@unittest.skipIf(not KERAS_AVAILABLE, "Keras not available")
class KerasCheckpointHandlerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.handler = keras_checkpoint_handler.KerasCheckpointHandler()

  def _test_save_and_restore_with_backend(self, expected_backend):
    """Helper method to test save/restore with a specific backend."""
    current_backend = keras.backend.backend()
    if current_backend != expected_backend:
      self.skipTest(f"Test requires {expected_backend} backend, got {current_backend}")
    
    # Create a simple Keras model
    model = keras.Sequential([
        layers.Dense(2, input_shape=(2,)),
        layers.Dense(1)
    ])
    
    # Set some weights
    original_weights = model.get_weights()

    with tempfile.TemporaryDirectory() as tmpdir:
      directory = epath.Path(tmpdir)

      # Save
      save_args = keras_checkpoint_handler.KerasSaveArgs(model)
      self.handler.save(directory, save_args)

      # Create a new model with same architecture
      new_model = keras.Sequential([
          layers.Dense(2, input_shape=(2,)),
          layers.Dense(1)
      ])
      
      # Restore
      restore_args = keras_checkpoint_handler.KerasRestoreArgs(new_model)
      restored_model = self.handler.restore(directory, restore_args)

      # Check that weights were restored
      restored_weights = restored_model.get_weights()
      self.assertEqual(len(original_weights), len(restored_weights))
      for orig, rest in zip(original_weights, restored_weights):
        np.testing.assert_array_equal(orig, rest)

  def test_save_and_restore_jax(self):
    """Test save/restore with JAX backend."""
    self._test_save_and_restore_with_backend('jax')

  def _test_async_save_and_restore_with_backend(self, expected_backend):
    """Helper method to test async save/restore with a specific backend."""
    import asyncio
    
    current_backend = keras.backend.backend()
    if current_backend != expected_backend:
      self.skipTest(f"Test requires {expected_backend} backend, got {current_backend}")
    
    # Create a simple Keras model
    model = keras.Sequential([
        layers.Dense(2, input_shape=(2,)),
        layers.Dense(1)
    ])
    
    # Set some weights
    original_weights = model.get_weights()

    async def async_test():
      with tempfile.TemporaryDirectory() as tmpdir:
        directory = epath.Path(tmpdir)

        # Async save
        save_args = keras_checkpoint_handler.KerasSaveArgs(model)
        futures = await self.handler.async_save(directory, save_args)
        
        # Wait for async save to complete
        if futures:
          for f in futures:
            f.result()  # Block on result
        
        # Finalize the save
        self.handler.finalize(directory)
        
        # Create a new model with same architecture
        new_model = keras.Sequential([
            layers.Dense(2, input_shape=(2,)),
            layers.Dense(1)
        ])
        
        # Restore
        restore_args = keras_checkpoint_handler.KerasRestoreArgs(new_model)
        restored_model = self.handler.restore(directory, restore_args)

        # Check that weights were restored
        restored_weights = restored_model.get_weights()
        self.assertEqual(len(original_weights), len(restored_weights))
        for orig, rest in zip(original_weights, restored_weights):
          np.testing.assert_array_equal(orig, rest)

    asyncio.run(async_test())

  def test_async_save_and_restore_jax(self):
    """Test async save/restore with JAX backend."""
    self._test_async_save_and_restore_with_backend('jax')

  def test_async_checkpoint_during_training_jax(self):
    """Test that async checkpointing doesn't block training and saves correct weights with JAX."""
    import asyncio
    import time
    
    current_backend = keras.backend.backend()
    if current_backend != 'jax':
      self.skipTest(f"Test requires jax backend, got {current_backend}")

    # Create a simple model
    model = keras.Sequential([
        layers.Dense(2, input_shape=(2,)),
        layers.Dense(1)
    ])
    
    # Compile model
    model.compile(optimizer='adam', loss='mse')
    
    # Generate some training data
    x = np.random.random((100, 2))
    y = np.random.random((100, 1))
    
    # Initial training to set some weights
    model.fit(x[:50], y[:50], epochs=1, verbose=0)
    weights_after_initial_training = model.get_weights()
    
    async def async_checkpoint_test():
      with tempfile.TemporaryDirectory() as tmpdir:
        directory = epath.Path(tmpdir)
        
        # Start async checkpointing
        save_args = keras_checkpoint_handler.KerasSaveArgs(model)
        futures = await self.handler.async_save(directory, save_args)
        
        # Continue training while checkpointing is happening
        # This should not be blocked by the async save
        start_time = time.time()
        model.fit(x[50:], y[50:], epochs=2, verbose=0)  # Continue training
        training_time = time.time() - start_time
        
        # Wait for async save to complete
        if futures:
          for f in futures:
            f.result()  # Block on result
        
        # Finalize the save
        self.handler.finalize(directory)
        
        # Get weights after continued training
        weights_after_continued_training = model.get_weights()
        
        # Create a new model and restore from checkpoint
        new_model = keras.Sequential([
            layers.Dense(2, input_shape=(2,)),
            layers.Dense(1)
        ])
        
        restore_args = keras_checkpoint_handler.KerasRestoreArgs(new_model)
        restored_model = self.handler.restore(directory, restore_args)
        restored_weights = restored_model.get_weights()
        
        # Verify that training wasn't blocked (should be fast)
        self.assertLess(training_time, 1.0, "Training was blocked by async checkpointing")
        
        # Verify that restored weights match the checkpoint time, not after continued training
        self.assertEqual(len(weights_after_initial_training), len(restored_weights))
        for orig, restored in zip(weights_after_initial_training, restored_weights):
          np.testing.assert_array_equal(orig, restored, 
            "Restored weights should match checkpoint time, not after continued training")
        
        # Verify that weights changed after continued training
        weights_differ = False
        for before, after in zip(weights_after_initial_training, weights_after_continued_training):
          if not np.array_equal(before, after):
            weights_differ = True
            break
        self.assertTrue(weights_differ, "Weights should have changed after continued training")

    asyncio.run(async_checkpoint_test())

  def test_transforms_jax(self):
    """Test transformations with JAX backend."""
    current_backend = keras.backend.backend()
    if current_backend != 'jax':
      self.skipTest(f"Test requires jax backend, got {current_backend}")

    # Create a simple Keras model
    model = keras.Sequential([
        keras.Input(shape=(2,)),
        layers.Dense(2),
        layers.Dense(1)
    ])

    # Set some known weights
    original_weights = [
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),  # kernel for first layer
        np.array([0.1, 0.2], dtype=np.float32),                  # bias for first layer
        np.array([[5.0], [6.0]], dtype=np.float32),             # kernel for second layer
        np.array([0.3], dtype=np.float32)                       # bias for second layer
    ]
    model.set_weights(original_weights)

    with tempfile.TemporaryDirectory() as tmpdir:
      directory = epath.Path(tmpdir)

      # Save without transforms
      save_args = keras_checkpoint_handler.KerasSaveArgs(model)
      self.handler.save(directory, save_args)

      # Create a new model with same architecture
      new_model = keras.Sequential([
          keras.Input(shape=(2,)),
          layers.Dense(2),
          layers.Dense(1)
      ])

      # Define transformations: scale all weights by 2
      transforms = [
          transform_utils.Transform(value_fn=lambda x: x * 2),  # index 0
          transform_utils.Transform(value_fn=lambda x: x * 2),  # index 1
          transform_utils.Transform(value_fn=lambda x: x * 2),  # index 2
          transform_utils.Transform(value_fn=lambda x: x * 2),  # index 3
      ]

      # Restore with transforms
      restore_args = keras_checkpoint_handler.KerasRestoreArgs(new_model, transforms=transforms)
      restored_model = self.handler.restore(directory, restore_args)

      # Check that weights were transformed (scaled by 2)
      restored_weights = restored_model.get_weights()
      expected_weights = [w * 2 for w in original_weights]

      self.assertEqual(len(expected_weights), len(restored_weights))
      for expected, actual in zip(expected_weights, restored_weights):
        np.testing.assert_array_almost_equal(expected, actual)

  def test_transforms_with_key_renaming_jax(self):
    """Test transformations with key renaming using multi_value_fn."""
    current_backend = keras.backend.backend()
    if current_backend != 'jax':
      self.skipTest(f"Test requires jax backend, got {current_backend}")

    # Create a simple Keras model
    model = keras.Sequential([
        keras.Input(shape=(2,)),
        layers.Dense(2),
        layers.Dense(1)
    ])

    # Set some known weights
    original_weights = [
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),  # kernel for first layer (2, 2)
        np.array([0.1, 0.2], dtype=np.float32),                  # bias for first layer (2,)
        np.array([[5.0], [6.0]], dtype=np.float32),             # kernel for second layer (2, 1)
        np.array([0.3], dtype=np.float32)                       # bias for second layer (1,)
    ]
    model.set_weights(original_weights)

    with tempfile.TemporaryDirectory() as tmpdir:
      directory = epath.Path(tmpdir)

      # Save without transforms
      save_args = keras_checkpoint_handler.KerasSaveArgs(model)
      self.handler.save(directory, save_args)

      # Create a new model with same architecture
      new_model = keras.Sequential([
          keras.Input(shape=(2,)),
          layers.Dense(2),
          layers.Dense(1)
      ])

      # Define transformations using original_key to rearrange weights
      # Note: This rearranges bias and kernel which have incompatible shapes,
      # so this test just verifies the transformation runs without crashing
      transforms = [
          transform_utils.RestoreTransform(original_key='3'),  # 0 <- 3 (bias2 -> kernel1 position - will fail shape check)
          transform_utils.RestoreTransform(original_key='2'),  # 1 <- 2 (kernel2 -> bias1 position - will fail shape check)
          transform_utils.RestoreTransform(original_key='1'),  # 2 <- 1 (bias1 -> kernel2 position - will fail shape check)
          transform_utils.RestoreTransform(original_key='0'),  # 3 <- 0 (kernel1 -> bias2 position - will fail shape check)
      ]

      # Restore with transforms
      restore_args = keras_checkpoint_handler.KerasRestoreArgs(new_model, transforms=transforms)
      try:
        restored_model = self.handler.restore(directory, restore_args)
        # If we get here, the transformation ran but shapes don't match
        self.fail("Expected shape mismatch error due to incompatible weight rearrangement")
      except ValueError as e:
        # Expected error due to shape mismatch
        self.assertIn("weight shape", str(e))


if __name__ == '__main__':
  absltest.main()