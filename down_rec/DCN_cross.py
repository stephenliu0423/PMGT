import torch
import torch.nn as nn
from PMGT_v2.config_file.param_rec import args


class DeepNet(nn.Module):
    """
    Deep part of Cross and Deep Network
    All of the layer in this module are full-connection layers
    """

    def __init__(self, input_feature_num, deep_layer: list):
        """
        :param input_feature_num: total num of input_feature, including of the embedding feature and dense feature
        :param deep_layer: a list contains the num of each hidden layer's units
        """
        super(DeepNet, self).__init__()
        fc_layer_list = [nn.Linear(input_feature_num, deep_layer[0]), nn.BatchNorm1d(deep_layer[0], affine=False),
                         nn.ReLU(inplace=True)]
        for i in range(1, len(deep_layer)):
            fc_layer_list.append(nn.Linear(deep_layer[i - 1], deep_layer[i]))
            fc_layer_list.append(nn.BatchNorm1d(deep_layer[i], affine=False))
            fc_layer_list.append(nn.ReLU(inplace=True))
        self.deep = nn.Sequential(*fc_layer_list)

    def forward(self, x):
        dense_output = self.deep(x)
        return dense_output


class CrossNet(nn.Module):
    """
    Cross layer part in Cross and Deep Network
    The ops in this module is x_0 * x_l^T * w_l + x_l + b_l for each layer l, and x_0 is the init input of this module
    """

    def __init__(self, input_feature_num, cross_layer: int):
        """
        :param input_feature_num: total num of input_feature, including of the embedding feature and dense feature
        :param cross_layer: the number of layer in this module expect of init op
        """
        super(CrossNet, self).__init__()
        self.cross_layer = cross_layer + 1  # add the first calculate
        weight_w = []
        weight_b = []
        batchnorm = []
        for i in range(self.cross_layer):
            weight_w.append(nn.Parameter(
                torch.nn.init.normal_(torch.empty(input_feature_num))))
            weight_b.append(nn.Parameter(
                torch.nn.init.normal_(torch.empty(input_feature_num))))
            batchnorm.append(nn.BatchNorm1d(input_feature_num, affine=False))
        self.weight_w = nn.ParameterList(weight_w)
        self.weight_b = nn.ParameterList(weight_b)
        self.batchnorm = nn.ModuleList(batchnorm)

    def forward(self, x):
        output = x
        x = x.reshape(x.shape[0], -1, 1)
        for i in range(self.cross_layer):
            output = torch.matmul(torch.bmm(x, torch.transpose(output.reshape(
                output.shape[0], -1, 1), 1, 2)), self.weight_w[i]) + self.weight_b[i] + output
            output = self.batchnorm[i](output)
        return output


class CDNet(nn.Module):
    """
    Cross and Deep Network in Deep & Cross Network for Ad Click Predictions
    """

    def __init__(self, num_users, num_items, pretrain_features):
        super(CDNet, self).__init__()
        self.user_num = num_users
        self.item_num = num_items
        deep_layer = eval(args.deep_layer)

        self.user_embedding = nn.Embedding(num_users, args.dim)
        self.input_dim = 2 * args.dim
        self._init_item_em(pretrain_features)

        # self.batchnorm = nn.BatchNorm1d(input_feature_num, affine=False)
        self.batcnnorm = nn.BatchNorm1d(self.input_dim, affine=False)
        self.CrossNet = CrossNet(self.input_dim, args.cross_layer_num)
        self.DeepNet = DeepNet(self.input_dim, deep_layer)
        last_layer_feature_num = self.input_dim + \
            deep_layer[-1]  # the dim of feature in last layer
        self.output_layer = nn.Linear(
            last_layer_feature_num, 1)  # 0, 1 classification
        nn.init.normal_(self.user_embedding.weight, std=0.01)

    def _init_item_em(self, pretrain_features):
        if args.pretrain:
            self.feat_item = nn.Embedding(self.item_num, pretrain_features.shape[-1])
            self.feat_item.weight.data.copy_(torch.from_numpy(pretrain_features))
            # self.feat_item.weight.requires_grad = False
            # self.embed_item_init = nn.Embedding(self.item_num, args.dim)
            # nn.init.normal_(self.embed_item_init.weight, std=0.01)
            # self.input_dim += pretrain_features.shape[-1]
        else:
            self.embed_item_init = nn.Embedding(self.item_num, args.dim)
            nn.init.normal_(self.embed_item_init.weight, std=0.01)

    def get_item_em(self, item_id):
        if args.pretrain:
            item_em = self.feat_item(item_id)
            # item_em = torch.cat((self.embed_item_init(item_id), self.feat_item(item_id)), -1)
        else:
            item_em = self.embed_item_init(item_id)
        return item_em

    def compute_score(self, feature):
        feature = self.batcnnorm(feature)
        out_cross = self.CrossNet(feature)
        out_deep = self.DeepNet(feature)
        final_feature = torch.cat((out_cross, out_deep), dim=-1)
        score = self.output_layer(final_feature).squeeze()
        score = torch.sigmoid(score)
        return score

    def forward(self, user_nodes, item_nodes):
        user_tensor = self.user_embedding(user_nodes)
        item_tensor = self.get_item_em(item_nodes)
        feature = torch.cat((user_tensor, item_tensor), -1)
        scores = self.compute_score(feature)
        return scores

