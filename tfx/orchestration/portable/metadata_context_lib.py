# Lint as: python2, python3
# Copyright 2020 Google LLC. All Rights Reserved.
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
"""TFX ml metadata library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import List, Text, Union

import absl
import tensorflow as tf

from ml_metadata.proto import metadata_store_pb2
from tfx.orchestration import metadata
from tfx.orchestration.portable import common_metadata_utils
from tfx.proto.orchestration import pipeline_pb2

CONTEXT_TYPE_EXECUTION_CACHE = 'execution_cache'
PType = Union[int, float, Text]


def _register_context_type_if_not_exist(
    metadata_handler: metadata.Metadata,
    context_spec: pipeline_pb2.ContextSpec) -> int:
  """Registers a context type if not exist, otherwise returns existing one.

  Args:
    metadata_handler: A handler to access MLMD store.
    context_spec: A pipeline_pb2.ContextSpec message that instructs registering
      of a context.

  Returns:
    id of the desired context type.
  """
  context_type = context_spec.type
  for k, v in context_spec.properties.items():
    context_type.properties[k] = common_metadata_utils.get_metadata_value_type(
        v)

  return metadata_handler.store.put_context_type(
      context_type, can_add_fields=True)


def _generate_context_proto(
    metadata_handler: metadata.Metadata,
    context_spec: pipeline_pb2.ContextSpec) -> metadata_store_pb2.Context:
  """Generates metadata_pb2.Context based on the ContextSpec message.

  Args:
    metadata_handler: A handler to access MLMD store.
    context_spec: A pipeline_pb2.ContextSpec message that instructs registering
      of a context.

  Returns:
    A metadata_store_pb2.Context message.
  """
  context_type_id = _register_context_type_if_not_exist(
      metadata_handler=metadata_handler, context_spec=context_spec)
  context = metadata_store_pb2.Context(
      type_id=context_type_id,
      name=common_metadata_utils.get_value(context_spec.name))
  for k, v in context_spec.properties.items():
    common_metadata_utils.set_metadata_value(context.properties[k], v)
  return context


def _register_context_if_not_exist(
    metadata_handler: metadata.Metadata,
    context_spec: pipeline_pb2.ContextSpec,
) -> metadata_store_pb2.Context:
  """Registers a context if not exist, otherwise returns the existing one.

  Args:
    metadata_handler: A handler to access MLMD store.
    context_spec: A pipeline_pb2.ContextSpec message that instructs registering
      of a context.

  Returns:
    An MLMD context.
  """
  context = _generate_context_proto(
      metadata_handler=metadata_handler, context_spec=context_spec)
  try:
    [context_id] = metadata_handler.store.put_contexts([context])
    context.id = context_id
  except tf.errors.AlreadyExistsError:
    context_name = common_metadata_utils.get_value(context_spec.name)
    absl.logging.debug('Context %s already exists.', context_name)
    context = metadata_handler.store.get_context_by_type_and_name(
        type_name=context_spec.type.name, context_name=context_name)
    assert context is not None, 'Context is missing for %s.' % (
        context_spec.name)

  absl.logging.debug('ID of context %s is %s.', context_spec, context.id)
  return context


def register_context_if_not_exists(
    metadata_handler: metadata.Metadata,
    context_type_name: Text,
    context_name: Text,
) -> metadata_store_pb2.Context:
  """Registers a context if not exist, otherwise returns the existing one.

  This is a simplified wrapper around the method above which only takes context
  type and context name.

  Args:
    metadata_handler: A handler to access MLMD store.
    context_type_name: The name of the context type.
    context_name: The name of the context.

  Returns:
    An MLMD context.
  """
  context_spec = pipeline_pb2.ContextSpec(
      name=pipeline_pb2.Value(
          field_value=metadata_store_pb2.Value(string_value=context_name)),
      type=metadata_store_pb2.ContextType(name=context_type_name))
  return _register_context_if_not_exist(
      metadata_handler=metadata_handler, context_spec=context_spec)


def register_contexts_if_not_exists(
    metadata_handler: metadata.Metadata,
    node_contexts: pipeline_pb2.NodeContexts,
) -> List[metadata_store_pb2.Context]:
  """Creates or fetches the contexts given specification.

  Args:
    metadata_handler: A handler to access MLMD store.
    node_contexts: A pipeline_pb2.NodeContext message that instructs registering
      of the contexts.

  Returns:
    A list of metadata_store_pb2.Context messages.
  """

  return [
      _register_context_if_not_exist(
          metadata_handler=metadata_handler, context_spec=context_spec)
      for context_spec in node_contexts.contexts
  ]
