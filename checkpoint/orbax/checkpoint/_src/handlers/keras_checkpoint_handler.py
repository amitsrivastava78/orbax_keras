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

"""KerasCheckpointHandler class for saving/loading Keras models."""

from __future__ import annotations

import dataclasses
from typing import Any, Optional, Union

from absl import logging
from etils import epath
import numpy as np
from orbax.checkpoint import checkpoint_args
from orbax.checkpoint import transform_utils
from orbax.checkpoint.checkpoint_args import register_with_handler
from orbax.checkpoint._src.handlers import async_checkpoint_handler
from orbax.checkpoint._src.handlers import checkpoint_handler
from orbax.checkpoint._src.handlers import pytree_checkpoint_handler
from orbax.checkpoint._src.metadata import tree as tree_metadata
from orbax.checkpoint._src.serialization import type_handlers
from orbax.checkpoint._src.tree import utils as tree_utils

try:
  import keras
except ImportError:
  keras = None


@dataclasses.dataclass
class KerasSaveArgs(checkpoint_args.CheckpointArgs):
  """Arguments for saving Keras models."""
  item: Any


@dataclasses.dataclass
class KerasRestoreArgs(checkpoint_args.CheckpointArgs):
  """Arguments for restoring Keras models."""
  item: Any
  transforms: Optional[Any] = None


class KerasCheckpointHandler(async_checkpoint_handler.AsyncCheckpointHandler):
  """Handler for saving and restoring Keras models."""

  def __init__(self):
    self._pytree_handler = None

  def save(self, directory: epath.Path, args: KerasSaveArgs) -> Any:
    """Saves a Keras model to the given directory."""
    if keras is None:
      raise ImportError("Keras is required for KerasCheckpointHandler")
    
    model = args.item
    if not isinstance(model, keras.Model):
      raise ValueError(f"Expected a Keras model, got {type(model)}")

    # Get the backend
    backend = keras.backend.backend()
    if backend in ['jax', 'tensorflow', 'torch']:
      # For all backends, use PyTreeCheckpointHandler to save model weights
      # Keras 3.0 get_weights() returns numpy arrays regardless of backend
      self._pytree_handler = pytree_checkpoint_handler.PyTreeCheckpointHandler()
      weights = model.get_weights()
      self._pytree_handler.save(directory, item=weights)
      self._pytree_handler.finalize(directory)
      return None
    else:
      raise ValueError(f"Unsupported backend: {backend}")

  async def async_save(self, directory: epath.Path, args: KerasSaveArgs) -> Optional[Any]:
    """Async save."""
    if keras is None:
      raise ImportError("Keras is required for KerasCheckpointHandler")
    
    model = args.item
    if not isinstance(model, keras.Model):
      raise ValueError(f"Expected a Keras model, got {type(model)}")

    # Get the backend
    backend = keras.backend.backend()
    if backend in ['jax', 'tensorflow', 'torch']:
      # For all backends, use PyTreeCheckpointHandler to save model weights
      # Keras 3.0 get_weights() returns numpy arrays regardless of backend
      self._pytree_handler = pytree_checkpoint_handler.PyTreeCheckpointHandler()
      weights = model.get_weights()
      futures = await self._pytree_handler.async_save(directory, item=weights)
      return futures
    else:
      raise ValueError(f"Unsupported backend: {backend}")

  def restore(self, directory: epath.Path, args: KerasRestoreArgs) -> Any:
    """Restores a Keras model from the given directory."""
    if keras is None:
      raise ImportError("Keras is required for KerasCheckpointHandler")
    
    backend = keras.backend.backend()
    if backend in ['jax', 'tensorflow', 'torch']:
      # For all backends, use PyTreeCheckpointHandler to restore weights
      # Keras 3.0 set_weights() accepts numpy arrays regardless of backend
      pytree_handler = pytree_checkpoint_handler.PyTreeCheckpointHandler()
      if args.transforms is not None:
        # If transforms are provided, we need to specify the target structure
        # Use the model's current weights as the target structure for transforms
        target_weights = args.item.get_weights()
        # Also need to provide restore_args matching the target structure
        restore_args = [type_handlers.RestoreArgs() for _ in target_weights]
        restored_weights = pytree_handler.restore(directory, item=target_weights, transforms=args.transforms, restore_args=restore_args)
      else:
        restored_weights = pytree_handler.restore(directory)
      # Set weights on the model
      args.item.set_weights(restored_weights)
      return args.item
    else:
      raise ValueError(f"Unsupported backend: {backend}")

  def metadata(self, directory: epath.Path) -> Optional[tree_metadata.TreeMetadata]:
    """Return metadata."""
    return super().metadata(directory)

  def finalize(self, directory: epath.Path) -> None:
    """Finalize save."""
    if self._pytree_handler is not None:
      self._pytree_handler.finalize(directory)
    super().finalize(directory)

  def close(self):
    """Close handler."""
    super().close()


# Apply decorators after class definition
KerasSaveArgs = register_with_handler(
    KerasCheckpointHandler, for_save=True
)(KerasSaveArgs)

KerasRestoreArgs = register_with_handler(
    KerasCheckpointHandler, for_restore=True
)(KerasRestoreArgs)