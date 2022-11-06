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
"""Classes for encoding features to embeddings.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from base_encoder import BaseFeatureEncoder
from config_file.config_PMGT import FLAGS
from utils.utils_bert import *

EMB_PT_SIZE = 128 * 512


class AttentionEncoder(BaseFeatureEncoder):
    def __init__(self,
                 use_input_bn=True,
                 act=None,
                 need_dense=True,
                 ps_hosts=None,
                 name=''):
        self._output_dim = FLAGS.hidden_dim
        self._feature_num = FLAGS.features_num
        self._use_input_bn = use_input_bn
        self._need_dense = need_dense
        self._name = name

        self._emb_table = {}
        self._act = act

    def encode(self, input_attrs):
        """Encode input_attrs to embeddings.

        Args:
        input_attrs: A list in the format of [continuous_attrs, categorical_attrs]

        Returns:
        Embeddings.W
        """
        continuous_attrs = input_attrs

        to_concats_con = None
        continuous_feats_num = self._feature_num

        if continuous_feats_num > 0:  # contains continuous features
            # to_concats_con = tf.log(
            #     tf.reshape(continuous_attrs, [-1, continuous_feats_num]) + 2)  # log operation
            to_concats_con = tf.reshape(continuous_attrs, [-1, continuous_feats_num])

        vis_feature = to_concats_con[:, :FLAGS.vis_features_num]
        text_feature = to_concats_con[:, FLAGS.vis_features_num:]
        with tf.variable_scope(self._name + 'attrs_attention_encoding', reuse=tf.AUTO_REUSE):
            if self._use_input_bn:
                vis_feature = tf.nn.l2_normalize(vis_feature, dim=1, name='vis_l2_normalize')
                text_feature = tf.nn.l2_normalize(text_feature, dim=1, name='text_l2_normalize')
                # vis_feature = tf.layers.batch_normalization(vis_feature, training=True)
                # text_feature = tf.layers.batch_normalization(text_feature, training=True)
            trans_vis = tf.layers.dense(vis_feature, self._output_dim, name='vis_trans')
            trans_text = tf.layers.dense(text_feature, self._output_dim, name='text_trans')
            concat_em = tf.tanh(tf.concat((trans_vis, trans_text), -1))
            gate_score = tf.layers.dense(concat_em, 2, activation=tf.nn.softmax, name='gate_score')  # (batch, 1)
            trans_feat = tf.concat([trans_vis[:, tf.newaxis, :], trans_text[:, tf.newaxis, :]], axis=1)  # (batch, 3, dim)
            raw_emb = tf.squeeze(tf.matmul(gate_score[:, tf.newaxis, :], trans_feat), [1])  # (batch_dim)
        return raw_emb, to_concats_con

