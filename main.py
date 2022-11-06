from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np

import random
import tensorflow as tf
from collections import Counter

from sklearn.metrics import roc_auc_score

from PMGT_v2.config_file.config_PMGT import get_graph_config
from trainer import LocalTFTrainer
from config_file.config_PMGT import FLAGS
import graphlearn as gl


rng = random.Random(FLAGS.seed)


def load_graph(categorical_attrs_desc=''):
    print('starting load graph...')
    # node_table, node_table_train, node_table_test, edge_table_org = FLAGS.tables.split(',')
    categorical_attrs_num = len(categorical_attrs_desc)
    continuous_attrs_num = FLAGS.features_num - categorical_attrs_num
    node_encoder = gl.Decoder(attr_types=["float"] * continuous_attrs_num, attr_delimiter=',')
    edge_encoder = gl.Decoder(weighted=True, attr_types=["int"], attr_delimiter=',')
    graph = gl.Graph().node(FLAGS.root_path + FLAGS.data_type + '/node_feat.txt', node_type="entity", decoder=node_encoder) \
        .node(FLAGS.root_path + FLAGS.data_type + '/node_feat_train.txt', node_type="entity", decoder=node_encoder, mask=gl.Mask.TRAIN) \
        .node(FLAGS.root_path + FLAGS.data_type + '/node_feat_test.txt', node_type="entity", decoder=node_encoder, mask=gl.Mask.TEST) \
        .edge(FLAGS.root_path + FLAGS.data_type + '/edge_org.txt', edge_type=("entity", "entity", "hrt"), decoder=edge_encoder, directed=False)
    return graph

def query(graph, mask):
    prefix = ('train', 'test', 'val')[mask.value - 1]
    bs = FLAGS.batch_size
    if prefix == 'train':
        print("train")
        seed = graph.V("entity", mask=mask).batch(bs).shuffle(traverse=True).alias(prefix + '_node')
    else:
        seed = graph.V("entity", mask=mask).batch(bs).alias(prefix + '_node')

    seed_edge = seed.outE("hrt").sample(1).by("random").alias(prefix + '_edge')
    seed_dst = seed_edge.inV().alias(prefix + '_pos')
    seed_neg = seed.outNeg("hrt").sample(1).by("random").alias(prefix + '_neg')

    current_hop_list = [seed, seed_dst, seed_neg]
    alias_prefix = [prefix+'_node', prefix+'_pos', prefix+'_neg']
    for idx, hop in enumerate(eval(FLAGS.hops_list)):
        next_hop_list = []
        for hop_q, alias_pre in zip(current_hop_list, alias_prefix):
            _alias = alias_pre + '_hop_' + str(idx + 1)
            # print(_alias)

            next_hop_list.append(hop_q.outV("hrt").sample(hop).by("edge_weight").alias(_alias))
        current_hop_list = next_hop_list
    return seed

def MCNSampling(graph, res, alias_prefix, mask=True, load_emb=False):

    sample_nodes = [res[alias_prefix + "_hop_" + str(idx + 1)] for idx in range(FLAGS.hops_num)]
    batch_size = res[alias_prefix].ids.shape[0]
    hops_nodes = np.hstack([np.tile(neigh_nodes.ids.reshape(batch_size, -1), (1, FLAGS.hops_num - i))
                            for i, neigh_nodes in enumerate(sample_nodes)])
    mask_length = int(FLAGS.neigh_num * FLAGS.mask_prob)
    neighbor_node = -1 * np.zeros((batch_size, FLAGS.neigh_num + mask_length))
    mask_positions = -1 * np.zeros((batch_size, mask_length))
    for i, neigh in enumerate(hops_nodes):
        neigh_count = Counter(neigh)
        node_count = neigh_count.most_common(FLAGS.neigh_num)
        node_list = np.array([node[0] for node in node_count])
        if len(node_list) < FLAGS.neigh_num:
            # node_list = np.append(node_list, -1 * np.ones(self.neigh_num - len(node_list)))
            node_list = np.append(node_list, np.random.choice(
                node_list, FLAGS.neigh_num - len(node_list), replace=True))
        mask_position, replace_index = generate_mask_index(node_list, mask_length)
        mask_index = node_list[mask_position]
        if mask:
            node_list[mask_position] = replace_index
        node_list = np.append(node_list, mask_index)
        neighbor_node[i, :] = node_list
        mask_positions[i, :] = mask_position
    neighbor_node = neighbor_node.astype(int)
    mask_positions = mask_positions.astype(int)
    neighbor_node = graph.get_nodes("entity", neighbor_node).float_attrs  # (batch, seq_length)
    ego_node = res[alias_prefix].float_attrs
    if load_emb:
        return res[alias_prefix].ids, ego_node, neighbor_node, mask_positions

    return ego_node, neighbor_node, mask_positions

def generate_mask_index(node_list, mask_length):
    cand_indexes = np.arange(len(node_list))
    rng.shuffle(cand_indexes)
    num_to_predict = min(FLAGS.neigh_num, max(1, mask_length))
    vocab_nodes = np.arange(FLAGS.num_nodes)
    masked_lms = []
    for index in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        # 80% of the time, replace with [MASK]
        if rng.random() < 0.8:
            masked_token = -1
        else:
            # 10% of the time, keep original
            if rng.random() < 0.5:
                masked_token = node_list[index]
            # 10% of the time, replace with random word
            else:
                masked_token = vocab_nodes[rng.randint(0, FLAGS.num_nodes - 1)]
        masked_lms.append([index, masked_token])
    assert len(masked_lms) <= num_to_predict
    masked_lms = sorted(masked_lms, key=lambda x: x[0])
    masked_position = np.array([x[0] for x in masked_lms])
    masked_token = np.array([x[1] for x in masked_lms])

    return masked_position, masked_token

def get_entity_embedding(graph):
    print("strating load node embedding")

    trainer = LocalTFTrainer(graph,
                             epoch=1,
                             optimizer=None)
    emb_out_op = trainer.model.reload_node_embedding()
    trainer.init()
    ckpt = tf.train.get_checkpoint_state(FLAGS.root_path + FLAGS.data_type + '/checkpoint')
    print(ckpt.all_model_checkpoint_paths)
    trainer.saver.restore(trainer.sess, ckpt.all_model_checkpoint_paths[-1])

    emb_seed = graph.V("entity").batch(FLAGS.batch_size).alias('emb_node')

    current_hop_list = [emb_seed]
    alias_prefix = ['emb_node']
    for idx, hop in enumerate(eval(FLAGS.hops_list)):
        next_hop_list = []
        for hop_q, alias_pre in zip(current_hop_list, alias_prefix):
            _alias = alias_pre + '_hop_' + str(idx + 1)
            print(_alias)

            next_hop_list.append(hop_q.outV("hrt").sample(hop).by("edge_weight").alias(_alias))
        current_hop_list = next_hop_list

    emb_ds = gl.Dataset(emb_seed.values())
    iter = 0
    sample_length = 0
    item_id = []
    item_content = []
    while True:
        try:
            res = emb_ds.next()

            src_ego_ids, src_ego_emb, src_nbr_emb, src_mask = MCNSampling(graph, res, alias_prefix="emb_node", load_emb=True)
            feed_dict = {trainer.model.src_ego_emb: src_ego_emb, trainer.model.src_nbr_emb: src_nbr_emb,
                         trainer.model.src_mask: src_mask,
                         trainer.model.hidden_dropout_prob: 0,
                         trainer.model.attention_probs_dropout_prob: 0}
            emb_out = trainer.sess.run(emb_out_op, feed_dict=feed_dict)

            ids_np = src_ego_ids.reshape([-1])
            sample_length += len(ids_np)
            for i in range(len(ids_np)):
                emb_np_s = emb_out[i]
                item_id.append(int(ids_np[i]))
                item_content.append(emb_np_s)
            iter += 1
        except gl.OutOfRangeError:
            break

    print('sample length:', sample_length)
    print('shape of item_id:', np.array(item_id).shape)
    print('shape of item_content:', np.array(item_content).shape)
    np.save(FLAGS.root_path + FLAGS.data_type + '/item_id.npy', np.array(item_id))
    np.save(FLAGS.root_path + FLAGS.data_type + '/item_feature.npy', np.array(item_content))
    print('finish...')


def train_and_evaluate(graph, seed, trainer, mask):
    loss, train_op, nsl_loss_, frs_loss_, logits, labels = trainer.model.build()
    train_ops = [train_op, loss, nsl_loss_, frs_loss_, logits, labels]
    trainer.init()
    print('training...')

    prefix = ('train', 'test', 'val')[mask.value - 1]
    ds = gl.Dataset(seed.values())
    epoch = FLAGS.epochs
    for idx in range(epoch):
        print('starting training epochs')
        total_loss = []
        nsl_loss = []
        frs_loss = []
        total_auc = []
        dur = []
        step = 0
        while True:
            try:
                res = ds.next()
                start_time = time.time()
                print("-----total epoch:", epoch, "epoch",idx, "step:", step,"------")

                src_ego_emb, src_nbr_emb, src_mask = MCNSampling(graph, res, alias_prefix=prefix + "_node")
                dst_ego_emb, dst_nbr_emb, dst_mask = MCNSampling(graph, res, alias_prefix=prefix + "_pos")
                neg_ego_emb, neg_nbr_emb, neg_mask = MCNSampling(graph, res, alias_prefix=prefix + "_neg")
                bs = src_ego_emb.shape[0]
                dst_ego_emb = dst_ego_emb.reshape(bs, -1)
                neg_ego_emb = neg_ego_emb.reshape(bs, -1)
                feed_dict = {trainer.model.src_ego_emb: src_ego_emb, trainer.model.src_nbr_emb: src_nbr_emb,
                             trainer.model.src_mask: src_mask,
                             trainer.model.dst_ego_emb: dst_ego_emb, trainer.model.dst_nbr_emb: dst_nbr_emb,
                             trainer.model.dst_mask: dst_mask,
                             trainer.model.neg_ego_emb: neg_ego_emb, trainer.model.neg_nbr_emb: neg_nbr_emb,
                             trainer.model.neg_mask: neg_mask,
                             trainer.model.hidden_dropout_prob: FLAGS.hidden_dropout_prob,
                             trainer.model.attention_probs_dropout_prob: FLAGS.attention_probs_dropout_prob}

                outs = trainer.sess.run(train_ops, feed_dict=feed_dict)
                end_time = time.time()
                total_loss.append(outs[1])
                nsl_loss.append(outs[2])
                frs_loss.append(outs[3])
                total_auc.append(roc_auc_score(y_score=outs[4], y_true=outs[5]))
                iter_time = end_time - start_time
                dur.append(iter_time)
                step += 1

            except gl.OutOfRangeError:
                break
        print("Epoch {:02d}, Time(s) {:.4f}, Loss {:.5f}, NSL_Loss {:.5f}, FRS_LOSS {:.5f}".
              format(idx, np.sum(dur), np.mean(total_loss), np.mean(nsl_loss), np.mean(frs_loss)))
        print("AUC {:.5f}".
              format(np.mean(total_auc)))
        if (idx + 1) % 2 == 0:
            print('saving model...')
            trainer.saver.save(trainer.sess, FLAGS.root_path + FLAGS.data_type + '/checkpoint/model.ckpt',
                               global_step=idx + 1)
        # evaluate_once(graph, trainer, train_ops, idx)
    evaluate_once(graph, trainer, train_ops, idx)

def evaluate_once(graph, trainer, train_ops, idx):
    ######## test each epoch
    test_seed = query(graph, gl.Mask.TEST)
    prefix = ('train', 'test', 'val')[gl.Mask.TEST.value - 1]
    test_ds = gl.Dataset(test_seed.values())
    iter = 0
    total_auc = []
    logits, labels = train_ops[-2:]

    while True:
        try:
            res = test_ds.next()
            print("-----evaluate once,", "step:", iter, "------")

            src_ego_emb, src_nbr_emb, src_mask = MCNSampling(graph, res, alias_prefix=prefix + "_node")
            dst_ego_emb, dst_nbr_emb, dst_mask = MCNSampling(graph, res, alias_prefix=prefix + "_pos")
            neg_ego_emb, neg_nbr_emb, neg_mask = MCNSampling(graph, res, alias_prefix=prefix + "_neg")
            bs = src_ego_emb.shape[0]
            dst_ego_emb = dst_ego_emb.reshape(bs, -1)
            neg_ego_emb = neg_ego_emb.reshape(bs, -1)
            feed_dict = {trainer.model.src_ego_emb: src_ego_emb, trainer.model.src_nbr_emb: src_nbr_emb,
                         trainer.model.src_mask: src_mask,
                         trainer.model.dst_ego_emb: dst_ego_emb, trainer.model.dst_nbr_emb: dst_nbr_emb,
                         trainer.model.dst_mask: dst_mask,
                         trainer.model.neg_ego_emb: neg_ego_emb, trainer.model.neg_nbr_emb: neg_nbr_emb,
                         trainer.model.neg_mask: neg_mask,
                         trainer.model.hidden_dropout_prob: 0,
                         trainer.model.attention_probs_dropout_prob: 0}
            logits_np, labels_np = trainer.sess.run([logits, labels], feed_dict=feed_dict)
            if len(labels_np) == 0:
                continue
            else:
                total_auc.append(roc_auc_score(y_score=logits_np, y_true=labels_np))
            iter += 1
        except gl.OutOfRangeError:
            break

    if len(total_auc) == 0:
        print('have not data for auc.')
    else:
        print('Testing epoch {}, AUC is: {:.4f}'.format(idx, np.mean(total_auc)))

def main():
    g = load_graph()
    g.init()
    if FLAGS.is_train:
        trainer = LocalTFTrainer(g,
                                 epoch=FLAGS.epochs,
                                 optimizer=None)

        seed = query(g, gl.Mask.TRAIN)
        train_and_evaluate(g, seed, trainer, gl.Mask.TRAIN)
    else:
        get_entity_embedding(g)


if __name__ == "__main__":
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    tf.set_random_seed(FLAGS.seed)
    config = get_graph_config(FLAGS.data_type)
    FLAGS.num_nodes = config['num_items']
    main()
