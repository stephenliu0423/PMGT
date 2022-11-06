import tensorflow as tf


def del_all_flags(FLAGS):
    for keys in [keys for keys in FLAGS._flags()]:
        FLAGS.__delattr__(keys)
del_all_flags(tf.flags.FLAGS)

flags = tf.app.flags

flags.DEFINE_integer('batch_size', 256, 'batch size of training data')
flags.DEFINE_integer('epochs', 7, 'epochs of training')
flags.DEFINE_string('learning_algo', 'adamW', 'method of optimization')
flags.DEFINE_integer('num_hidden_layers', 2, 'number of hidden layer')
flags.DEFINE_float('initializer_range', 0.01, 'range of initializer')
flags.DEFINE_integer('num_attention_heads', 2, 'number of attention heads')

flags.DEFINE_integer('hidden_dim', 128, 'dimension of hidden layer of model')
flags.DEFINE_integer('intermediate_dim', 64, 'dimension of intermediate layer of model')  # 128
flags.DEFINE_float('learning_rate', 0.0001, 'learning rate of training')
flags.DEFINE_float('weight_decay', 0.0001, 'weight of param decay')
flags.DEFINE_float('hidden_dropout_prob', 0, 'rate of dropout for hidden layer')
flags.DEFINE_float('attention_probs_dropout_prob', 0, 'rate of dropout for attention weights')

flags.DEFINE_integer('neigh_num', 10, 'number of neighbors of nodes in graph')  # 10
flags.DEFINE_integer('features_num', 2304, 'number of features')
flags.DEFINE_integer('vis_features_num', 1536, 'number of visual features')
flags.DEFINE_integer('text_features_num', 768, 'number of text features')
# flags.DEFINE_integer('audio_features_num', 128, 'number of audio features')  # 128
flags.DEFINE_integer('neg_num', 5, 'number of negative samples')
flags.DEFINE_integer('hops_num', 3, 'number of hops of random walk')
flags.DEFINE_string('hops_list', '[10,5,2]', 'list of node number in each hop')  # [10,5,2]
flags.DEFINE_float('alpha', 0.2, 'weights of similarity attetion')

flags.DEFINE_integer('num_train', 5000, 'number of training steps')
flags.DEFINE_integer('warmup_step', 500, 'number of warmup steps')
flags.DEFINE_float('beta', 1, 'weights of feature reconstruct loss')

flags.DEFINE_integer('num_nodes', 0, 'number of nodes in graph')
flags.DEFINE_float('mask_prob', 0.2, 'probs of mask')
flags.DEFINE_integer('is_train', 0, 'whether to train')
flags.DEFINE_integer('seed', 2020, 'seed of random')
flags.DEFINE_string('root_path', 'data/', 'path of root')
flags.DEFINE_string('data_type', 'toys', 'type of dataset')
flags.DEFINE_integer('save_embedding', 0, 'whether to save embedding')

flags.DEFINE_integer("task_index", None, "Task index")
flags.DEFINE_integer("task_count", None, 'Task count')
flags.DEFINE_string("job_name", None, "worker or ps")
flags.DEFINE_string("ps_hosts", "", "ps hosts")
flags.DEFINE_string("worker_hosts", "", "worker hosts")
FLAGS = flags.FLAGS


def get_graph_config(data_type):
    if data_type == 'video':  # yerars <= 2015 threshold>=3
        kge_config = {
            'num_items': 7252,
            'num_triples': 88606*2
        }
    elif data_type == 'toys':  # years <=2015 threshold>=4
        kge_config = {
            'num_items': 6451,
            'num_triples': 15363*2
        }
    elif data_type == 'tools':  # year2 <= 2015 threshold>=3
        kge_config = {
            'num_items': 5982,
            'num_triples': 12927*2
        }
    elif data_type == 'movie_lens':  # time <=12e+08  threshold>=3 user_list>=20 rating>=4
        kge_config = {
            'num_items': 000,
            'num_triples': 000
        }
    else:
        raise RuntimeError('please input right data type')
    return kge_config
