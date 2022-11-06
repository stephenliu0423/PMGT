import numpy as np
from PMGT_v2.config_file.param_rec import args
from PMGT_v2.utils.utils import read_local_pkl
from PMGT_v2.config_file.config_downtream import get_rec_config
from torch.utils.data import Dataset
try:
    import common_io
except:
    pass


class RecData(Dataset):
    def __init__(self):
        self.data_path = args.root_path + args.data_type
        self.rec_path = self.data_path + '/downstream/'
        self.num_items = get_rec_config(args.data_type)['num_items']
        print('number of users:', get_rec_config(args.data_type)['num_users'])
        print('number of items:', self.num_items)
        self.all_items = set(range(self.num_items))
        self.read_pkl = read_local_pkl
        self.train_dic, self.train_pairs = self.read_train_data()
        self.pretrain_features = self.read_pretrain_data()

    def read_train_data(self):
        train_pairs = []
        train_dic = self.read_pkl(self.rec_path + 'train_rec.pickle')
        for user, pos_items in train_dic.items():
            for item in pos_items:
                train_pairs.append((user, item))
        return train_dic, train_pairs

    def read_features_table(self):
        item_id = np.load(self.data_path + '/item_id.npy')
        features = np.load(self.data_path + '/item_feature.npy')
        index = np.argsort(item_id)
        print(np.array(item_id)[index])
        features = np.array(features)[index, :]
        print('shape of features:', features.shape)
        return features

    def read_pretrain_data(self):
        if args.pretrain:
            features = self.read_features_table()
            features = features[:self.num_items]
        else:
            features = None
        return features

    def __getitem__(self, index):
        user = self.train_pairs[index][0]
        pos_item = self.train_pairs[index][1]
        neg_item = np.random.randint(self.num_items)
        while neg_item in self.train_dic[user]:
            neg_item = np.random.randint(self.num_items)
        return user, pos_item, neg_item

    def __len__(self):
        return len(self.train_pairs)

    def read_test_data(self):
        test_dic = self.read_pkl(self.rec_path + 'test_rec.pickle')
        for user, pos_items in test_dic.items():
            neg_items = self.all_items - pos_items
            if user in self.train_dic:
                neg_items -= self.train_dic[user]
            neg_items = np.array(list(neg_items))
            if len(neg_items) > 1000:
                neg_items = np.random.choice(neg_items, 1000)
            yield user, np.array(list(pos_items)), neg_items
