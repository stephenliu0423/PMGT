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
"""Classes used to construct EgoTensor encoders."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from MModal_encoder import AttentionEncoder
from PMGT_model import PMGT_Model
from utils.loss_fn import gather_indexes, similarity_loss_split
from utils.optimization import create_optimizer
from base_encoder import BaseGraphEncoder
from utils.utils_bert import *
from config_file.config_PMGT import FLAGS


class EgoGraph_Encoder(BaseGraphEncoder):
    def __init__(self,
                 neigh_num=12,
                 token_type_vocab_size=2,
                 use_postion_embeddings=True,
                 use_token_role=True,
                 position_embedding_name='position_embeddings',
                 token_type_embedding_name='token_type_embeddings',
                 max_position_embeddings=128):

        self.neigh_num = neigh_num
        self.use_token_role = use_token_role
        self.use_position_embedding = use_postion_embeddings
        if use_token_role:
            self.token_role_table = tf.get_variable(
                name=token_type_embedding_name,
                shape=[token_type_vocab_size, FLAGS.hidden_dim],
                initializer=create_initializer(FLAGS.initializer_range))
        if use_postion_embeddings: #use position
            self.full_position_embeddings = tf.get_variable(
                name=position_embedding_name,
                shape=[max_position_embeddings, FLAGS.hidden_dim],
                initializer=create_initializer(FLAGS.initializer_range))
        categorical_attrs_desc = ''
        self.hidden_dropout_prob = tf.placeholder(tf.float32, shape=None, name='drop_hidden_layer')
        self.attention_probs_dropout_prob = tf.placeholder(tf.float32, shape=None, name='drop_attention')
        self._dropout = self.hidden_dropout_prob
        self._feature_encoders = [AttentionEncoder(ps_hosts=FLAGS.ps_hosts)] * 2
        self.bert_model = PMGT_Model(self.hidden_dropout_prob, self.attention_probs_dropout_prob)
        self.src_ego_emb = tf.placeholder(tf.float32, shape=(None, FLAGS.features_num), name='src_ego_emb')
        self.src_nbr_emb = tf.placeholder(tf.float32, shape=(None, neigh_num, FLAGS.features_num), name='src_nbr_emb')
        self.dst_ego_emb = tf.placeholder(tf.float32, shape=(None, FLAGS.features_num), name='dst_ego_emb')
        self.dst_nbr_emb = tf.placeholder(tf.float32, shape=(None, neigh_num, FLAGS.features_num), name='pos_nbr_emb')
        self.neg_ego_emb = tf.placeholder(tf.float32, shape=(None, FLAGS.features_num), name='neg_ego_emb')
        self.neg_nbr_emb = tf.placeholder(tf.float32, shape=(None, neigh_num, FLAGS.features_num), name='neg_nbr_emb')
        self.src_mask = tf.placeholder(tf.int32, shape=(None, 2), name='src_mask')
        self.dst_mask = tf.placeholder(tf.int32, shape=(None, 2), name='dst_mask')
        self.neg_mask = tf.placeholder(tf.int32, shape=(None, 2), name='neg_mask')


    def _forward(self, inputs, input_mask=None, head_mask=None):
        all_encoder_layers, sequence_output, pooled_output = self.bert_model.encode(inputs, input_mask, head_mask)
        return [all_encoder_layers, sequence_output, pooled_output]

    def add_side_em(self, head_em, head_neigh_em):
        shape = get_shape_list(head_em)
        hidden_size = shape[2]
        head_seq_em = tf.concat([head_em, tf.reshape(head_neigh_em, [-1, FLAGS.neigh_num, hidden_size])],
                                1)  # (batch, neigh_num+1, feat)
        if self.use_token_role:
            token_role_id = np.zeros(FLAGS.neigh_num + 1)
            token_role_id[0] = 1
            role_em = tf.nn.embedding_lookup(self.token_role_table,
                                             token_role_id.astype(np.int64),
                                             name='embedding_look_up_token_role')  # (neigh_num+1, dim)
            role_em = role_em[tf.newaxis, :]  # (1, neigh_num+1, dim)
            head_seq_em += role_em

        if self.use_position_embedding:
            seq_len = get_shape_list(head_seq_em)[1]
            position_embeddings = tf.slice(self.full_position_embeddings, [0, 0],
                                           [seq_len, -1])[tf.newaxis, :]  # (seq_len, dim)
            head_seq_em += position_embeddings

        all_seq_em = head_seq_em
        all_seq_em = layer_norm_and_dropout(all_seq_em, self._dropout)
        return all_seq_em

    def encode(self, ego_center, ego_neighbor):
        # encode features.
        ego_em, ego_feat = self._feature_encoders[0].encode(ego_center)
        batch_size = get_shape_list(ego_em)[0]
        ego_em = tf.expand_dims(ego_em, 1)  # (batch, 1, dim)
        ego_neigh_em, ego_neigh_feat = self._feature_encoders[1].encode(ego_neighbor)  # (batch*(neigh_num+mg), dim)
        mask_length = int(self.neigh_num * FLAGS.mask_prob)
        ego_neigh_em = tf.reshape(ego_neigh_em, [-1, FLAGS.neigh_num + mask_length, FLAGS.hidden_dim])
        ego_neigh_em = ego_neigh_em[:, :FLAGS.neigh_num, :]

        ego_mask_feat = tf.reshape(ego_neigh_feat, [-1, FLAGS.neigh_num + mask_length, FLAGS.features_num])[:, FLAGS.neigh_num:, :]
        out_em = self.add_side_em(ego_em, ego_neigh_em)  # (batch, seq_len, dim)
        input_mask = None
        head_mask = None
        return self._forward(out_em, input_mask, head_mask), ego_mask_feat, ego_feat

    def feature_reconstruct_loss_split(self, output_emb, mask_feat, head_feat, mask_position, feat_num, L=2):
        mask_feat = tf.reshape(mask_feat, [-1, feat_num])
        with tf.variable_scope('reconstruct_loss_split', reuse=tf.AUTO_REUSE):
            output_emb = tf.layers.dense(
                output_emb,
                feat_num,
                kernel_initializer=create_initializer(FLAGS.initializer_range))
        mask_emb = gather_indexes(output_emb[:, 1:, :], mask_position)
        # head_emb = tf.reduce_mean(output_emb, axis=1)

        if L == 2:
            dis_mask = tf.reduce_mean(tf.square(mask_emb - mask_feat))
        else:
            dis_mask = tf.reduce_mean(tf.abs(mask_emb - mask_feat))
        return dis_mask

    def build(self):

        pos_src_tensors, pos_src_neigh_feat, pos_src_feat = self.encode(self.src_ego_emb, self.src_nbr_emb)
        pos_dst_tensors, pos_dst_neigh_feat, pos_dst_feat = self.encode(self.dst_ego_emb, self.dst_nbr_emb)
        neg_dst_tensors, neg_dst_neigh_feat, neg_dst_feat = self.encode(self.neg_ego_emb, self.neg_nbr_emb)


        self.nsl_loss, logits, labels = similarity_loss_split(
            pos_src_tensors[-1], pos_dst_tensors[-1], neg_dst_tensors[-1])  # input:(batch, out_dim)
        frs_loss_src = self.feature_reconstruct_loss_split(pos_src_tensors[1], pos_src_neigh_feat,
                                                      pos_src_feat,
                                                      self.src_mask, FLAGS.features_num)
        frs_loss_pos = self.feature_reconstruct_loss_split(pos_dst_tensors[1], pos_dst_neigh_feat,
                                                      pos_dst_feat,
                                                      self.dst_mask, FLAGS.features_num)
        frs_loss_neg = self.feature_reconstruct_loss_split(neg_dst_tensors[1], neg_dst_neigh_feat,
                                                      neg_dst_feat,
                                                      self.neg_mask, FLAGS.features_num)
        self.frs_loss = (frs_loss_neg + frs_loss_pos + frs_loss_src) / 3
        self.frs_loss = FLAGS.beta * self.frs_loss
        self.loss = self.nsl_loss + self.frs_loss

        self.train_op = create_optimizer(self.loss, FLAGS.learning_rate,
                                         FLAGS.weight_decay, FLAGS.num_train, FLAGS.warmup_step)

        with tf.name_scope('metrics'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar("nsl_loss", self.nsl_loss)
            tf.summary.scalar("frs_loss", self.frs_loss)

        self.summary_op = tf.summary.merge(tf.get_collection(
            tf.GraphKeys.SUMMARIES, scope='metrics'))
        print(self.loss, self.nsl_loss)
        return self.loss, self.train_op, self.nsl_loss, self.frs_loss, logits, labels

    def reload_node_embedding(self):
        emb, _, _ = self.encode(self.src_ego_emb, self.src_nbr_emb)
        out_put = emb[-1] / tf.sqrt(tf.reduce_sum(tf.square(emb[-1]), axis=-1, keep_dims=True))
        return out_put

