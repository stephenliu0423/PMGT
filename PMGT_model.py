# Copyright 2020 Alibaba Group Holding Limited. All Rights Reserved.
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
# =============================================================================
"""Class of EgoGraph Convolutional layer"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils.utils_bert import *

from config_file.config_PMGT import FLAGS


class PMGT_Model(object):
    def __init__(self, hidden_dropout_prob, attention_probs_dropout_prob):
        self.encoder = PMGTEncoder(hidden_dropout_prob, attention_probs_dropout_prob)
        self.num_hidden_layers = FLAGS.num_hidden_layers
        self.hidden_dim = FLAGS.hidden_dim
        self.initializer_range = FLAGS.initializer_range

    def encode(self, seq_vecs, input_mask=None, head_mask=None):
        with tf.variable_scope("bert", reuse=tf.AUTO_REUSE):
            input_shape = get_shape_list(seq_vecs)  # (batch, seq_length, hidden_dim)
            batch_size = input_shape[0]
            seq_length = input_shape[1]
            if input_mask is None:
                input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)
            if head_mask is None:
                head_mask = [None] * self.num_hidden_layers
            with tf.variable_scope('encoder'):
                attention_mask = create_attention_mask_from_input_mask(
                    seq_vecs, input_mask)  # (batch, seq_length, seq_length)
                all_encoder_layers = self.encoder.transformer(
                    seq_vecs, attention_mask=attention_mask, head_mask=head_mask)
            sequence_output = all_encoder_layers[-1]  # (batch, (neigh + 1), dim)
            with tf.variable_scope("pooler"):
                # pooled_output = tf.reduce_mean(sequence_output, axis=1)
                pooled_output = sequence_output[:, 0, :]
                pooled_output = tf.layers.dense(pooled_output,
                                                self.hidden_dim,
                                                # activation=tf.tanh,
                                                kernel_initializer=create_initializer(self.initializer_range))
        return all_encoder_layers, sequence_output, pooled_output


class PMGTEncoder(object):
    def __init__(self, hidden_dropout_prob, attention_probs_dropout_prob):
        super(PMGTEncoder, self).__init__()
        self.hidden_dim = FLAGS.hidden_dim
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.num_attention_heads = FLAGS.num_attention_heads
        if self.hidden_dim % self.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (self.hidden_dim, self.num_attention_heads))
        self.attention_head_size = int(self.hidden_dim / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.intermediate_size = FLAGS.intermediate_dim
        self.num_hidden_layers = FLAGS.num_hidden_layers

    def transformer(self, input_tensor, attention_mask, head_mask, do_return_all_layers=True):
        input_shape = get_shape_list(input_tensor)  # (batch, seq_length, hidden_dim)
        prev_output = input_tensor
        all_layer_outputs = []
        for layer_idx in range(self.num_hidden_layers):
            with tf.variable_scope("layer_%d" % layer_idx):
                layer_input = prev_output
                with tf.variable_scope("attention"):
                    attention_head = self.attention_layer(layer_input, layer_input,
                                                          attention_mask, head_mask[layer_idx],
                                                          initializer_range=FLAGS.initializer_range)
                layer_output = self.trans_layer(attention_head,
                                                initializer_range=FLAGS.initializer_range)
                prev_output = layer_output
                all_layer_outputs.append(layer_output)

        if do_return_all_layers:
            final_outputs = []
            for layer_output in all_layer_outputs:
                final_output = reshape_from_matrix(layer_output, input_shape)
                final_outputs.append(final_output)
            return final_outputs
        else:
            final_output = reshape_from_matrix(prev_output, input_shape)
            return final_output

    def transpose_for_scores(self, input_tensor, batch_size, seq_length):
        shape = [batch_size, seq_length, self.num_attention_heads, self.attention_head_size]
        output_tensor = tf.reshape(input_tensor, shape)
        output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor

    def attention_layer(self, from_tensor, to_tensor, attention_mask=None, head_mask=None,
                        query_act=None, key_act=None, value_act=None, initializer_range=0.02):
        with tf.variable_scope("self"):
            from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
            to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])
            from_seq_length = from_shape[1]
            to_seq_length = to_shape[1]
            batch_size = from_shape[0]
            if len(from_shape) != len(to_shape):
                raise ValueError(
                    "The rank of `from_tensor` must match the rank of `to_tensor`.")
            # Scalar dimensions referenced here:
            #   B = batch size (number of sequences)
            #   F = `from_tensor` sequence length
            #   T = `to_tensor` sequence length
            #   N = `num_attention_heads`
            #   H = `size_per_head`

            from_tensor_2d = reshape_to_matrix(from_tensor)  # (batch * sequence, hidden_dim)
            to_tensor_2d = reshape_to_matrix(to_tensor)

            # `query_layer` = [B*F, N*H]
            query_layer = tf.layers.dense(
                from_tensor_2d,
                self.all_head_size,
                activation=query_act,
                name="query",
                kernel_initializer=create_initializer(initializer_range))

            # `key_layer` = [B*T, N*H]
            key_layer = tf.layers.dense(
                to_tensor_2d,
                self.all_head_size,
                activation=key_act,
                name="key",
                kernel_initializer=create_initializer(initializer_range))

            # `value_layer` = [B*T, N*H]
            value_layer = tf.layers.dense(
                to_tensor_2d,
                self.all_head_size,
                activation=value_act,
                name="value",
                kernel_initializer=create_initializer(initializer_range))

            dis_layer = tf.layers.dense(
                to_tensor_2d,
                self.all_head_size,
                activation=value_act,
                name="distance",
                kernel_initializer=create_initializer(initializer_range)
            )

            # `query_layer` = [B, N, F, H]
            query_layer = self.transpose_for_scores(query_layer, batch_size, from_seq_length)

            # `key_layer` = [B, N, T, H]
            key_layer = self.transpose_for_scores(key_layer, batch_size, to_seq_length)

            # `value_layer` = [B, N, T, H]
            value_layer = self.transpose_for_scores(value_layer, batch_size, to_seq_length)

            # 'dis_layer' = [B, N, T, H]
            dis_layer = self.transpose_for_scores(dis_layer, batch_size, to_seq_length)
            # Take the dot product between "query" and "key" to get the raw
            # attention scores.
            # `attention_scores` = [B, N, F, T]
            dis_scores = self.ecu_distance(dis_layer, dis_layer)  # (batch, N, F, F)
            attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
            attention_scores = tf.multiply(attention_scores,
                                           1.0 / math.sqrt(float(self.attention_head_size)))

            if attention_mask is not None:
                # `attention_mask` = [B, 1, F, T]
                attention_mask = tf.expand_dims(attention_mask, axis=[1])

                # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
                # masked positions, this operation will create a tensor which is 0.0 for
                # positions we want to attend and -10000.0 for masked positions.
                adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

                # Since we are adding it to the raw scores before the softmax, this is
                # effectively the same as removing these entirely.
                attention_scores += adder

            # Normalize the attention scores to probabilities.
            # `attention_probs` = [B, N, F, T]
            # attention_scores = attention_scores * dis_scores
            attention_probs = tf.nn.softmax(attention_scores)
            attention_probs = (1 - FLAGS.alpha) * attention_probs + FLAGS.alpha * dis_scores
            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            attention_probs = dropout(attention_probs, self.attention_probs_dropout_prob)
            if head_mask is not None:
                attention_probs = attention_probs * head_mask

            # `context_layer` = [B, N, F, H]
            context_layer = tf.matmul(attention_probs, value_layer)

            # `context_layer` = [B, F, N, H]
            context_layer = tf.transpose(context_layer, [0, 2, 1, 3])
            context_shape = get_shape_list(context_layer)
            # `context_layer` = [B, F, N*H]
            context_layer = tf.reshape(
                context_layer,
                [context_shape[0], context_shape[1], self.all_head_size])

        with tf.variable_scope("attention_output"):
            attention_output = tf.layers.dense(
                context_layer,
                self.hidden_dim,
                kernel_initializer=create_initializer(initializer_range))
            attention_output = dropout(attention_output, self.hidden_dropout_prob)
            attention_output = layer_norm(attention_output + from_tensor)
        return attention_output

    def trans_layer(self, attention_output, intermediate_act_fn=gelu, initializer_range=0.02):
        with tf.variable_scope("intermediate"):
            intermediate_output = tf.layers.dense(attention_output,
                                                  self.intermediate_size,
                                                  activation=intermediate_act_fn,
                                                  kernel_initializer=create_initializer(initializer_range))

        # Down-project back to `hidden_size` then add the residual.
        with tf.variable_scope("layer_output"):
            layer_output = tf.layers.dense(
                intermediate_output,
                self.hidden_dim,
                kernel_initializer=create_initializer(initializer_range))
            layer_output = dropout(layer_output, self.hidden_dropout_prob)
            layer_output = layer_norm(layer_output + attention_output)
        return layer_output

    def ecu_distance(self, X, Y):
        X_l2 = tf.sqrt(tf.reduce_sum(tf.square(X), axis=-1))  # (batch, N, seq_length)
        Y_l2 = tf.sqrt(tf.reduce_sum(tf.square(Y), axis=-1))  # (batch, N, seq_length)
        X_Y = tf.matmul(X, Y, transpose_b=True)  # (batch, N, seq_length, seq_length)
        dis_score = 1 - X_Y / (tf.expand_dims(X_l2, 3) * tf.expand_dims(Y_l2, 2))
        self_score = tf.ones_like(X_l2)  # (batch, N, seq)
        dis_score += tf.matrix_diag(self_score)  # (batch, N, seq, seq)
        dis_score = tf.nn.softmax(dis_score)  # (batch, N, seq, seq)
        return dis_score

