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
"""Prepressing using tensorflow_text BertTokenizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Text

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

from tensorflow.python.eager.context import eager_mode  # pylint: disable=g-direct-tensorflow-import


_CLS = '[CLS]'
_PAD = '[PAD]'
_SEP = '[SEP]'


class BertPreprocessor(object):
  """Bert Tokenizer built ontop of tensorflow_text.BertTokenizer."""

  def __init__(self, model_link: Text):
    self._model_link = model_link
    self._model = hub.KerasLayer(model_link)
    self._find_special_tokens()

  def _find_special_tokens(self):
    """Find the special token ID's for [CLS] [PAD] [SEP].

    Since each Bert model is trained on different vocabulary, it's important
    to find the special token indices pertaining to that model.
    Since in Transform, tensorflow_hub.KerasLayer loads a symbolic tensor, turn
    on eager mode to get the actual vocab_file location.
    """

    with eager_mode():
      model = hub.KerasLayer(self._model_link)
      vocab = model.resolved_object.vocab_file.asset_path.numpy()
      self._do_lower_case = model.resolved_object.do_lower_case.numpy()
      with tf.io.gfile.GFile(vocab, 'r') as f:
        lines = f.read().split('\n')
        self._sep_id = lines.index(_SEP)
        self._cls_id = lines.index(_CLS)
        self._pad_id = lines.index(_PAD)

  def tokenize_single_sentence_unpad(self,
                                     sequence: tf.Tensor,
                                     max_len: int = 128,
                                     add_cls: bool = True,
                                     add_sep: bool = True):
    """Tokenize a sentence with the BERT model vocab file and without padding.

    Add special tokens according to config.

    Args:
      sequence: Tensor of shape [batch_size, 1].
      max_len: The number of tokens after padding and truncating.
      add_cls: Whether to add CLS token at the front of each sequence.
      add_sep: Whether to add SEP token at the end of each sequence.

    Returns:
      word_ids: Ragged tokenized sequences [batch_size, None].
    """
    vocab_file_path = self._model.resolved_object.vocab_file.asset_path
    tokenizer = text.BertTokenizer(
        vocab_file_path,
        lower_case=self._do_lower_case,
        token_out_type=tf.int64)
    word_ids = tokenizer.tokenize(sequence)
    # Tokenizer default puts tokens into array of size 1. merge_dims flattens it
    word_ids = word_ids.merge_dims(-2, -1)
    if add_cls:
      cls_token = tf.fill([tf.shape(sequence)[0], 1],
                          tf.constant(self._cls_id, dtype=tf.int64))

      word_ids = tf.concat([cls_token, word_ids], 1)

    if add_sep:
      sep_token = tf.fill([tf.shape(sequence)[0], 1],
                          tf.constant(self._sep_id, dtype=tf.int64))

      word_ids = word_ids[:, :max_len - 1]
      word_ids = tf.concat([word_ids, sep_token], 1)

    return word_ids

  def tokenize_single_sentence_pad(self,
                                   sequence: tf.Tensor,
                                   max_len: int = 128,
                                   add_cls: bool = True,
                                   add_sep: bool = True):
    """Tokenize a single sentence according to the vocab used by the Bert model.

    Add special tokens according to config.

    Args:
      sequence: Tensor of shape [batch_size, 1].
      max_len: The number of tokens after padding and truncating.
      add_cls: Whether to add CLS token at the front of each sequence.
      add_sep: Whether to add SEP token at the end of each sequence.

    Returns:
      word_ids: Tokenized sequences [batch_size, max_len].
      input_mask: Mask padded tokens [batch_size, max_len].
      segment_ids: Distinguish multiple sequences [batch_size, max_len].
    """
    word_ids = self.tokenize_single_sentence_unpad(sequence, max_len, add_cls,
                                                   add_sep)

    word_ids = word_ids.to_tensor(
        shape=[None, max_len],
        default_value=tf.constant(self._pad_id, dtype=tf.int64))

    input_mask = tf.cast(tf.not_equal(word_ids, self._pad_id), tf.int64)
    segment_ids = tf.fill(tf.shape(input_mask), tf.constant(0, dtype=tf.int64))

    return word_ids, input_mask, segment_ids

  def tokenize_sentence_pair(self, sequence_a: tf.Tensor, sequence_b: tf.Tensor,
                             max_len: int):
    """Tokenize a sequence pair.

    Tokenize each sequence with self.tokenize_single_sentence. Then add CLS
    token in front of the first sequence, add SEP tokens between the two
    sequences and at the end of the second sequence.

    Args:
      sequence_a: [batch_size, 1]
      sequence_b: [batch_size, 1]
      max_len: The length of the concatenated tokenized sentences.

    Returns:
      word_ids: Tokenized sequences [batch_size, max_len].
      input_mask: Mask padded tokens [batch_size, max_len].
      segment_ids: Distinguish multiple sequences [batch_size, max_len].
    """
    # TODO(dzats): the issue here is nuanced. Depending on the dataset, one
    # might want to keep the entire first sentence, or the second. Consider
    # alternate truncate stratagies.
    sentence_len = max_len // 2
    word_id_a = self.tokenize_single_sentence_unpad(
        sequence_a,
        sentence_len,
        True,
        True,
    )

    word_id_b = self.tokenize_single_sentence_unpad(
        sequence_b,
        sentence_len,
        False,
        True,
    )

    word_ids = tf.concat([word_id_a, word_id_b], 1)
    word_ids = word_ids.to_tensor(
        shape=[None, max_len],
        default_value=tf.constant(self._pad_id, dtype=tf.int64))

    input_mask = tf.cast(tf.not_equal(word_ids, self._pad_id), tf.int64)
    # Fill a ragged tensor of zero with word_id_a's shape
    segment_ids = tf.cast(word_id_a < 0, tf.int64)
    segment_ids = segment_ids.to_tensor(
        shape=[None, max_len], default_value=tf.constant(1, dtype=tf.int64))
    return word_ids, input_mask, segment_ids

  def char_index_to_bert_index(self, char_index: tf.Tensor,
                               original_sentence: tf.Tensor,
                               tokenized_sentence: tf.RaggedTensor,
                               shifts: int = 0):

    @tf.function
    def space_seperate_helper(x):
      s, c = x[0], x[1]
      comb = (tf.constant(0), tf.constant(0))
      condition = lambda i, cur: tf.less(i, c)
      def action(i, cur):
        char = tf.strings.substr(s, tf.squeeze(i), 1)
        return tf.cond(tf.equal(char, ' '), lambda: (i+1, cur+1),
            lambda: (i+1, cur))
      return tf.while_loop(condition, action, comb)[1]

    space_seperate_result = tf.map_fn(space_seperate_helper,
                                      (original_sentence, char_index),
                                      dtype=tf.int32)

    @tf.function
    def find_bert_index(x):
      index, rag = x[0], x[1]
      comb = (tf.constant(0), tf.constant(0))
      # account for special tokens
      condition = lambda i, cur: tf.less(i, index+shifts)
      def action(i, cur):
        return (i+1, cur + tf.size(rag[i]))
      return tf.while_loop(condition, action, comb)[1]

    answer = tf.map_fn(find_bert_index,
                       (space_seperate_result, tokenized_sentence),
                       dtype=tf.int32)

    return answer

  def preprocess_squad(self, max_len: int, context: tf.Tensor,
      question: tf.Tensor, answer_start: tf.SparseTensor,
      answer_text: tf.SparseTensor):
    """Preprocess the squad dataset.

    Concatenate the context and question tensors. Tokenize them. Convert answer
    to vector tensors indicating the start and end token of to the answer in the
    context input.

    Args:
      max_len: The length of the concatenated tokenized sentences.
      context: [batch_size, 1]
      question: [batch_size, 1]
      answer_start: [batch_size,]
      answer_text: [batch_size,]

    Returns:
      word_ids: Tokenized sequences [batch_size, max_len].
      input_mask: Mask padded tokens [batch_size, max_len].
      segment_ids: Distinguish multiple sequences [batch_size, max_len].
      labels: Dim two tokens [batch_size, max_len, 2]
    """
  

    # answer_start and answer_text are sparse tensor. Currently, just use the
    # first possible answer.
    if tf.rank(answer_start) != 1:
      answer_start = tf.reshape(tf.slice(answer_start, [0, 0], [-1, 1]), [-1])
      answer_text = tf.reshape(tf.slice(answer_text, [0, 0], [-1, 1]), [-1])
    else:
      answer_start = tf.reshape(answer_start, [-1])
      answer_text = tf.reshape(answer_text, [-1])

    token_question = self.tokenize_single_sentence_unpad(
        question,
        None
    )

    token_context = self.tokenize_single_sentence_unpad(
        context,
        None,
        False,
        True
    )
    
    label_start = self.char_index_to_bert_index(
        answer_start,
        context,
        tf.cast(token_context, tf.int32),
    )

    label_start = tf.map_fn(
        lambda x: tf.size(x[0]) + x[1],
        (token_question, label_start),
        dtype=tf.int32)

    vocab_file_path = self._model.resolved_object.vocab_file.asset_path
    tokenizer = text.BertTokenizer(
        vocab_file_path,
        lower_case=self._do_lower_case,
        token_out_type=tf.int64)
    tokenized_answer_text = tokenizer.tokenize(answer_text)
    # Tokenizer default puts tokens into array of size 1. merge_dims flattens it
    tokenized_answer_text = tokenized_answer_text.merge_dims(-2, -1)
    answer_size = tf.map_fn(tf.size, tokenized_answer_text, dtype=tf.int32)
    # Account for 0 index
    label_end = label_start + answer_size - 1
    label_start_one = tf.reshape(
        tf.one_hot(label_start, max_len),
        [-1, max_len])
    label_end_one = tf.reshape(tf.one_hot(label_end, max_len), [-1, max_len])
    labels = tf.stack([label_start_one, label_end_one], axis=2)
    # Dealing with question and context
    word_ids = tf.concat([token_question, token_context], 1)
    word_ids = word_ids.to_tensor(
        shape=[None, max_len],
        default_value=tf.constant(self._pad_id, dtype=tf.int64))

    input_mask = tf.cast(tf.not_equal(word_ids, self._pad_id), tf.int64)
    # Fill a ragged tensor of zero with word_id_a's shape
    segment_ids = tf.cast(token_question < 0, tf.int64)
    segment_ids = segment_ids.to_tensor(
        shape=[None, max_len],
        default_value=tf.constant(1, dtype=tf.int64))

    return word_ids, input_mask, segment_ids, labels
