from PMGT_v2.config_file.param_rec import args
from PMGT_v2.utils.utils import read_local_pkl
from PMGT_v2.config_file.config_downtream import get_rec_config
from torch.utils.data import Dataset
import numpy as np


def read_features_table():
    num_items = get_rec_config(args.data_type)['num_items']
    item_id = np.load(args.root_path + args.data_type + '/item_id.npy')
    features = np.load(args.root_path + args.data_type + '/item_feature.npy')
    index = np.argsort(item_id)
    print(np.array(item_id)[index])
    features = np.array(features)[index, :]
    print('shape of features:', features.shape)
    features = features[:num_items, :]
    print('shape of features:', features.shape)
    return features


class RecData(Dataset):
    def __init__(self, data_name):
        self.data_path = args.root_path + args.data_type
        self.rec_path = self.data_path + '/downstream/'
        self.read_pkl = read_local_pkl
        self.table_reader = self.read_pkl(self.data_path + '/downstream/' + data_name)

    # transform data
    def convert_example(self, example):
        user_id = example[0]
        item_id = example[1]
        label = example[2]
        return int(user_id), int(item_id), int(label)

    def __getitem__(self, index):
        example = self.table_reader[index]
        example_transform = self.convert_example(example)
        return example_transform

    def __len__(self):
        return len(self.table_reader)


