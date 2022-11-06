import argparse
import random

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch table IO')

    parser.add_argument('--root_path', default='../data/')

    parser.add_argument('--data_type', default='tools', type=str, help='type of data')

    # dimension of model
    parser.add_argument('--dim', type=int, default=128, help='dim of graph nodes')

    # setting of model training
    parser.add_argument('--batch_size', type=int, default=512, help='size of batch of data')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning ratio')  # 0.001
    parser.add_argument('--l2_re', type=float, default=0, help='regularization cofficient')  # 0.0001
    parser.add_argument('--pretrain', type=int, default=1, help='whether pretrain')

    # setting of NCF model
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers of mlp')
    parser.add_argument('--dropout', type=float, default=0, help='rate of dropout')

    # setting of DCN model
    parser.add_argument('--deep_layer', type=str, default='[128, 64]', help='layer of deep part')
    parser.add_argument('--cross_layer_num', type=int, default=2, help='number of cross layers')

    # setting of testing
    parser.add_argument('--Ks', nargs='?', default='[5, 10, 20]', help='topK recommendation')
    parser.add_argument('--rep_flag', type=bool, default=True, help='whether present topK result')

    parser.add_argument('--useGPU', default=1, type=int, help='whether to use GPU')
    parser.add_argument('--seed', default=2019, type=int, help='random seed')
    # Parse the arguments.
    args_l = parser.parse_args()

    # Set seeds
    torch.manual_seed(args_l.seed)
    random.seed(args_l.seed)
    np.random.seed(args_l.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args_l.seed)
        torch.backends.cudnn.deterministic = True
        args_l.useGPU = 1
    return args_l


args = parse_args()

