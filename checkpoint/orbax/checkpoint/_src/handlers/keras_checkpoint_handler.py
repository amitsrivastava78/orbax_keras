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
from orbax.checkpoint import options as options_lib
from orbax.checkpoint import transform_utils
from orbax.checkpoint.checkpoint_args import register_with_handler
from orbax.checkpoint._src.handlers import async_checkpoint_handler
from orbax.checkpoint._src.handlers import checkpoint_handler
from orbax.checkpoint._src.handlers import pytree_checkpoint_handler
from orbax.checkpoint._src.handlers.pytree_checkpoint_handler import PyTreeSaveArgs
from orbax.checkpoint._src.metadata import tree as tree_metadata
from orbax.checkpoint._src.serialization import type_handlers
from orbax.checkpoint._src.tree import utils as tree_utils
from orbax.checkpoint._src.handlers import composite_checkpoint_handler
from orbax.checkpoint._src.handlers.composite_checkpoint_handler import CompositeArgs

try:
  import keras
except ImportError:
  keras = None


@dataclasses.dataclass
class KerasSaveArgs(checkpoint_args.CheckpointArgs):
  """Arguments for saving Keras models."""
  item: Any
  custom_metadata: Optional[Any] = None


@dataclasses.dataclass
class KerasRestoreArgs(checkpoint_args.CheckpointArgs):
  """Arguments for restoring Keras models."""
  item: Any
  transforms: Optional[Any] = None
  partial_restore: bool = False
  restore_args: Optional[Any] = None
  transforms_default_to_original: bool = True


class KerasCheckpointHandler(async_checkpoint_handler.AsyncCheckpointHandler):
  """Handler for saving and restoring Keras models."""

  def __init__(
      self,
      multiprocessing_options: options_lib.MultiprocessingOptions = options_lib.MultiprocessingOptions()
  ):
    """Creates KerasCheckpointHandler.

    Args:
      multiprocessing_options: Options for multiprocessing/multihost support.
        See orbax.checkpoint.options.MultiprocessingOptions for details.
    """
    self._multiprocessing_options = multiprocessing_options
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
      self._pytree_handler = pytree_checkpoint_handler.PyTreeCheckpointHandler(
          multiprocessing_options=self._multiprocessing_options
      )
      weights = model.get_weights()
      pytree_save_args = PyTreeSaveArgs(item=weights, custom_metadata=args.custom_metadata)
      self._pytree_handler.save(directory, args=pytree_save_args)
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
      self._pytree_handler = pytree_checkpoint_handler.PyTreeCheckpointHandler(
          multiprocessing_options=self._multiprocessing_options
      )
      weights = model.get_weights()
      pytree_save_args = PyTreeSaveArgs(item=weights, custom_metadata=args.custom_metadata)
      futures = await self._pytree_handler.async_save(directory, args=pytree_save_args)
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
      pytree_handler = pytree_checkpoint_handler.PyTreeCheckpointHandler(
          multiprocessing_options=self._multiprocessing_options
      )
      if args.transforms is not None:
        # If transforms are provided, we need to specify the target structure
        # Use the model's current weights as the target structure for transforms
        target_weights = args.item.get_weights()
        # Use provided restore_args or create default ones
        restore_args = args.restore_args
        if restore_args is None:
          restore_args = [type_handlers.RestoreArgs() for _ in target_weights]
        pytree_restore_args = pytree_checkpoint_handler.PyTreeRestoreArgs(
            item=target_weights, 
            transforms=args.transforms, 
            restore_args=restore_args,
            partial_restore=args.partial_restore,
            transforms_default_to_original=args.transforms_default_to_original
        )
        restored_weights = pytree_handler.restore(directory, args=pytree_restore_args)
      else:
        if args.partial_restore or args.restore_args is not None:
          # For partial restore or custom restore_args, we need to provide the target structure
          target_weights = args.item.get_weights()
          pytree_restore_args = pytree_checkpoint_handler.PyTreeRestoreArgs(
              item=target_weights,
              restore_args=args.restore_args,
              partial_restore=args.partial_restore,
              transforms_default_to_original=args.transforms_default_to_original
          )
          restored_weights = pytree_handler.restore(directory, args=pytree_restore_args)
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


@dataclasses.dataclass
class KerasCompositeSaveArgs(checkpoint_args.CheckpointArgs):
  """Arguments for saving multiple Keras objects in a composite checkpoint."""
  items: dict[str, Any]
  custom_metadata: Optional[dict[str, Any]] = None


@dataclasses.dataclass
class KerasCompositeRestoreArgs(checkpoint_args.CheckpointArgs):
  """Arguments for restoring multiple Keras objects from a composite checkpoint."""
  items: dict[str, Any]
  transforms: Optional[dict[str, Any]] = None
  partial_restore: bool = False
  restore_args: Optional[dict[str, Any]] = None
  transforms_default_to_original: bool = True


class KerasCompositeCheckpointHandler(composite_checkpoint_handler.CompositeCheckpointHandler):
  """Handler for saving and restoring multiple Keras objects in a single checkpoint.

  This handler allows saving/loading multiple heterogeneous objects (Keras models,
  optimizers, training state, custom metadata, etc.) in a single checkpoint operation.
  """

  def __init__(
      self,
      multiprocessing_options: options_lib.MultiprocessingOptions = options_lib.MultiprocessingOptions()
  ):
    """Creates KerasCompositeCheckpointHandler.

    Args:
      multiprocessing_options: Options for multiprocessing/multihost support.
        See orbax.checkpoint.options.MultiprocessingOptions for details.
    """
    # Create a handler registry and register the handlers we need
    from orbax.checkpoint._src.handlers import handler_registration
    from orbax.checkpoint._src.handlers.composite_checkpoint_handler import CompositeOptions
    
    handler_registry = handler_registration.DefaultCheckpointHandlerRegistry()
    
    # Register handlers for the args types we use
    handler_registry.add(None, KerasSaveArgs, KerasCheckpointHandler())
    handler_registry.add(None, KerasRestoreArgs, KerasCheckpointHandler())
    handler_registry.add(None, pytree_checkpoint_handler.PyTreeSaveArgs, pytree_checkpoint_handler.PyTreeCheckpointHandler())
    handler_registry.add(None, pytree_checkpoint_handler.PyTreeRestoreArgs, pytree_checkpoint_handler.PyTreeCheckpointHandler())

    composite_options = CompositeOptions(
        multiprocessing_options=multiprocessing_options,
        async_options=options_lib.AsyncOptions(create_directories_asynchronously=False)
    )

    super().__init__(
        composite_options=composite_options,
        handler_registry=handler_registry
    )

  def save(self, directory: epath.Path, args: KerasCompositeSaveArgs) -> Any:
    """Saves multiple Keras objects to the given directory."""
    if keras is None:
      raise ImportError("Keras is required for KerasCompositeCheckpointHandler")

    # Convert KerasCompositeSaveArgs to CompositeArgs
    composite_items = {}
    for key, item in args.items.items():
      if isinstance(item, keras.Model):
        # For Keras models, use KerasSaveArgs
        custom_meta = args.custom_metadata.get(key) if args.custom_metadata else None
        composite_items[key] = KerasSaveArgs(item=item, custom_metadata=custom_meta)
      else:
        # For other objects, use PyTreeSaveArgs
        custom_meta = args.custom_metadata.get(key) if args.custom_metadata else None
        composite_items[key] = PyTreeSaveArgs(item=item, custom_metadata=custom_meta)

    composite_args = CompositeArgs(**composite_items)
    result = super().save(directory, composite_args)
    self.finalize(directory)
    return result

  async def async_save(self, directory: epath.Path, *args, **kwargs) -> Optional[Any]:
    """Async save multiple Keras objects."""
    if keras is None:
      raise ImportError("Keras is required for KerasCompositeCheckpointHandler")
    
    # Handle different call patterns
    if args and isinstance(args[0], KerasCompositeSaveArgs):
      # Called directly from test with KerasCompositeSaveArgs
      save_args = args[0]
      # Convert KerasCompositeSaveArgs to CompositeArgs
      composite_items = {}
      for key, item in save_args.items.items():
        if isinstance(item, keras.Model):
          # For Keras models, use KerasSaveArgs
          custom_meta = save_args.custom_metadata.get(key) if save_args.custom_metadata else None
          composite_items[key] = KerasSaveArgs(item=item, custom_metadata=custom_meta)
        else:
          # For other objects, use PyTreeSaveArgs
          custom_meta = save_args.custom_metadata.get(key) if save_args.custom_metadata else None
          composite_items[key] = PyTreeSaveArgs(item=item, custom_metadata=custom_meta)

      composite_args = CompositeArgs(**composite_items)
      return await super().async_save(directory, composite_args)
    else:
      # Called from parent with CompositeArgs, delegate
      return await super().async_save(directory, *args, **kwargs)

  def restore(self, directory: epath.Path, args: KerasCompositeRestoreArgs) -> Any:
    """Restores multiple Keras objects from the given directory."""
    if keras is None:
      raise ImportError("Keras is required for KerasCompositeCheckpointHandler")

    # Convert KerasCompositeRestoreArgs to CompositeArgs
    composite_items = {}
    for key, item in args.items.items():
      if isinstance(item, keras.Model):
        # For Keras models, use KerasRestoreArgs
        transforms = args.transforms.get(key) if args.transforms else None
        restore_args_item = args.restore_args.get(key) if args.restore_args else None
        composite_items[key] = KerasRestoreArgs(
            item=item,
            transforms=transforms,
            partial_restore=args.partial_restore,
            restore_args=restore_args_item,
            transforms_default_to_original=args.transforms_default_to_original
        )
      else:
        # For other objects, use PyTreeRestoreArgs
        # If item is None, the PyTree handler will restore to saved structure
        transforms = args.transforms.get(key) if args.transforms else None
        restore_args_item = args.restore_args.get(key) if args.restore_args else None
        composite_items[key] = pytree_checkpoint_handler.PyTreeRestoreArgs(
            item=item,
            transforms=transforms,
            partial_restore=args.partial_restore,
            restore_args=restore_args_item,
            transforms_default_to_original=args.transforms_default_to_original
        )

    composite_args = CompositeArgs(**composite_items)
    restored_composite = super().restore(directory, composite_args)

    # Convert back to the expected format - return a dict with the restored objects
    result = {}
    for key in args.items.keys():
      result[key] = restored_composite[key]
    return result


# Apply decorators after class definition
KerasSaveArgs = register_with_handler(
    KerasCheckpointHandler, for_save=True
)(KerasSaveArgs)

KerasRestoreArgs = register_with_handler(
    KerasCheckpointHandler, for_restore=True
)(KerasRestoreArgs)

KerasCompositeSaveArgs = register_with_handler(
    KerasCompositeCheckpointHandler, for_save=True
)(KerasCompositeSaveArgs)

KerasCompositeRestoreArgs = register_with_handler(
    KerasCompositeCheckpointHandler, for_restore=True
)(KerasCompositeRestoreArgs)