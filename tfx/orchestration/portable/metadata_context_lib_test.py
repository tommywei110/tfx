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
"""Tests for tfx.orchestration.portable.mlmd_context_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Standard Imports
import tensorflow as tf
from ml_metadata.proto import metadata_store_pb2
from tfx.orchestration import metadata
from tfx.orchestration.portable import metadata_context_lib
from tfx.orchestration.portable import test_utils
from tfx.proto.orchestration import pipeline_pb2


class ContextMetadataUtilsTest(tf.test.TestCase):

  def setUp(self):
    super(ContextMetadataUtilsTest, self).setUp()
    self._connection_config = metadata_store_pb2.ConnectionConfig()
    self._connection_config.sqlite.SetInParent()

  def testRegisterContexts(self):
    node_contexts = pipeline_pb2.NodeContexts()
    test_utils.load_proto_from_text('node_context_spec.pbtxt', node_contexts)
    with metadata.Metadata(connection_config=self._connection_config) as m:
      metadata_context_lib.register_contexts_if_not_exists(
          metadata_handler=m, node_contexts=node_contexts)
      # Duplicated call should succeed.
      contexts = metadata_context_lib.register_contexts_if_not_exists(
          metadata_handler=m, node_contexts=node_contexts)

      self.assertProtoEquals(
          """
          id: 1
          name: 'my_context_type_one'
          properties {
            key: "property_a"
            value: INT
          }
          """, m.store.get_context_type('my_context_type_one'))
      self.assertProtoEquals(
          """
          id: 2
          name: 'my_context_type_two'
          properties {
            key: "property_a"
            value: INT
          }
          properties {
            key: "property_b"
            value: STRING
          }
          """, m.store.get_context_type('my_context_type_two'))
      self.assertEqual(
          contexts[0],
          m.store.get_context_by_type_and_name('my_context_type_one',
                                               'my_context_one'))
      self.assertEqual(
          contexts[1],
          m.store.get_context_by_type_and_name('my_context_type_one',
                                               'my_context_two'))
      self.assertEqual(
          contexts[2],
          m.store.get_context_by_type_and_name('my_context_type_two',
                                               'my_context_three'))
      self.assertEqual(contexts[0].properties['property_a'].int_value, 1)
      self.assertEqual(contexts[1].properties['property_a'].int_value, 2)
      self.assertEqual(contexts[2].properties['property_a'].int_value, 3)
      self.assertEqual(contexts[2].properties['property_b'].string_value, '4')

  def testRegisterContextByTypeAndName(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      metadata_context_lib.register_context_if_not_exists(
          metadata_handler=m,
          context_type_name='my_context_type',
          context_name='my_context')
      # Duplicated call should succeed.
      context = metadata_context_lib.register_context_if_not_exists(
          metadata_handler=m,
          context_type_name='my_context_type',
          context_name='my_context')

      self.assertProtoEquals(
          """
          id: 1
          name: 'my_context_type'
          """, m.store.get_context_type('my_context_type'))
      self.assertEqual(
          context,
          m.store.get_context_by_type_and_name('my_context_type', 'my_context'))


if __name__ == '__main__':
  tf.test.main()
