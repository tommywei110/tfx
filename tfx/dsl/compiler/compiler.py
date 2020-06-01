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
"""Compile a TFX pipeline or a component into a uDSL IR proto."""

# TODO(b/149535307): Remove __future__ imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import absl

from tfx.components.base import base_node
from tfx.components.base import executor_spec
from tfx.components.common_nodes import resolver_node
from tfx.dsl.experimental import latest_artifacts_resolver
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.orchestration import data_types
from tfx.orchestration import pipeline
from tfx.proto.orchestration import pipeline_pb2


PIPELINE_ROOT_PARAMETER_NAME = "pipeline_root"
PIPELINE_RUN_ID_PARAMETER_NAME = "pipeline_run_id"
PIPELINE_CONTEXT_TYPE_NAME = "pipeline"
PIPELINE_RUN_CONTEXT_TYPE_NAME = "pipeline_run"


class Compiler(object):
  """Compile a TFX pipeline or a component into a uDSL IR proto."""

  def __init__(self):
    self._pipeline = None
    self._component_pbs = {}

  def _set_up_runtime_parameters(
      self, pipeline_info: data_types.PipelineInfo
  ) -> pipeline_pb2.PipelineRuntimeSpec:
    """Helper function that builds runtime parameters from the pipeline info."""
    runtime_spec = pipeline_pb2.PipelineRuntimeSpec()

    pipeline_root_pb = runtime_spec.pipeline_root
    pipeline_root_pb.name = PIPELINE_ROOT_PARAMETER_NAME
    pipeline_root_pb.type = pipeline_pb2.RuntimeParameter.Type.STRING
    if pipeline_info.pipeline_root:
      pipeline_root_pb.default_value.string_value = pipeline_info.pipeline_root

    run_id_pb = runtime_spec.pipeline_run_id
    run_id_pb.name = PIPELINE_RUN_ID_PARAMETER_NAME
    run_id_pb.type = pipeline_pb2.RuntimeParameter.Type.STRING
    return runtime_spec

  def _compile_component(
      self,
      tfx_component: base_node.BaseNode
  ) -> pipeline_pb2.PipelineNode:
    """Compile an individual component into a PipelineNode proto.

    Args:
      tfx_component: A TFX component.

    Returns:
      A PipelineNode proto that encodes information of the component.
    """
    node = pipeline_pb2.PipelineNode()

    # Step 1: Node info
    node.node_info.type.name = tfx_component.type
    node.node_info.id = tfx_component.id

    # Step 2: Node Context
    pipeline_context_pb = node.contexts.contexts.add()
    pipeline_context_pb.type.name = PIPELINE_CONTEXT_TYPE_NAME
    pipeline_context_pb.name.field_value.string_value = self._pipeline.pipeline_info.pipeline_context_name
    # Resolver node does not have pipeline run context
    if resolver_node.RESOLVER_CLASS not in tfx_component.exec_properties:
      pipeline_run_context_pb = node.contexts.contexts.add()
      pipeline_run_context_pb.type.name = PIPELINE_RUN_CONTEXT_TYPE_NAME
      runtime_param = pipeline_run_context_pb.name.runtime_parameter
      runtime_param.name = PIPELINE_RUN_CONTEXT_TYPE_NAME
      runtime_param.type = pipeline_pb2.RuntimeParameter.Type.STRING

    # Step 3: Node inputs
    for key, value in tfx_component.inputs.items():
      input_spec = node.inputs.inputs[key]
      channel = input_spec.channels.add()
      if value.producer_component_id:
        channel.producer_node_query.id = value.producer_component_id

        producer_pb = self._component_pbs[value.producer_component_id]
        for producer_context in producer_pb.contexts.contexts:
          context_query = channel.context_queries.add()
          context_query.type.CopyFrom(producer_context.type)
          context_query.name.CopyFrom(producer_context.name)

      artifact_type = value.type._get_artifact_type()  # pylint: disable=protected-access
      channel.artifact_query.type.CopyFrom(artifact_type)

      if value.output_key:
        channel.output_key = value.output_key

    # Step 3.1: Special treatment for Resolver node
    if resolver_node.RESOLVER_CLASS in tfx_component.exec_properties:
      resolver = tfx_component.exec_properties[resolver_node.RESOLVER_CLASS]
      if resolver == latest_artifacts_resolver.LatestArtifactsResolver:
        node.inputs.resolver_config.resolver_policy = (
            pipeline_pb2.ResolverConfig.ResolverPolicy.LATEST_ARTIFACT)
      elif resolver == latest_blessed_model_resolver.LatestBlessedModelResolver:
        node.inputs.resolver_config.resolver_policy = (
            pipeline_pb2.ResolverConfig.ResolverPolicy.LATEST_BLESSED_MODEL)
      else:
        node.inputs.resolver_config.resolver_policy = (
            pipeline_pb2.ResolverConfig.ResolverPolicy
            .RESOLVER_POLICY_UNSPECIFIED)

    # Step 4: Node outputs
    for key, value in tfx_component.outputs.items():
      output_spec = node.outputs.outputs[key]
      artifact_type = value.type._get_artifact_type()  # pylint: disable=protected-access
      output_spec.artifact_spec.type.CopyFrom(artifact_type)

    # Step 5: Node parameters
    for key, value in tfx_component.exec_properties.items():
      if value is None:
        continue
      parameter_value = node.parameters.parameters[key]
      if isinstance(value, str):
        parameter_value.field_value.string_value = value
      elif isinstance(value, int):
        parameter_value.field_value.int_value = value
      elif isinstance(value, float):
        parameter_value.field_value.double_value = value
      else:
        absl.logging.warning(
            "Component {} got unsupported parameter {} with type {}.".format(
                tfx_component.id, key, type(value)))

    # Step 6: Executor
    if isinstance(tfx_component.executor_spec, executor_spec.ExecutorClassSpec):
      node.executor.python_class_executor_spec.class_path = tfx_component.executor_spec.class_path

    # Step 7: Upstream/Downstream nodes
    # Note: the order of tfx_component.upstream_nodes is inconsistent from
    # run to run. We sort them so that compiler generates consistent result.
    node.upstream_nodes.extend(sorted([
        upstream_component.id
        for upstream_component in tfx_component.upstream_nodes
    ]))
    node.downstream_nodes.extend(sorted([
        downstream_component.id
        for downstream_component in tfx_component.downstream_nodes
    ]))

    # Step 8: Node execution opitons
    # TBD waiting for SDK implementation

    return node

  def compile(self, tfx_pipeline: pipeline.Pipeline) -> pipeline_pb2.Pipeline:
    """Compile a tfx pipeline into uDSL proto.

    Args:
      tfx_pipeline: A TFX pipeline.

    Returns:
      A Pipeline proto that encodes all necessary information of the pipeline.
    """
    self._pipeline = tfx_pipeline
    pipeline_pb = pipeline_pb2.Pipeline()
    pipeline_pb.pipeline_info.id = self._pipeline.pipeline_info.pipeline_name
    pipeline_pb.runtime_spec.CopyFrom(
        self._set_up_runtime_parameters(self._pipeline.pipeline_info))

    self._component_pbs = {}
    for node in self._pipeline.components:
      component_pb = self._compile_component(node)
      pipeline_or_node = pipeline_pb.PipelineOrNode()
      pipeline_or_node.pipeline_node.CopyFrom(component_pb)
      pipeline_pb.nodes.append(pipeline_or_node)
      self._component_pbs[node.id] = component_pb

    # Currently only Synchrnous mode is supported
    pipeline_pb.execution_mode = pipeline_pb2.Pipeline.ExecutionMode.SYNC
    return pipeline_pb
