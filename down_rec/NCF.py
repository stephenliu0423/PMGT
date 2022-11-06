import torch
import torch.nn as nn
from PMGT_v2.config_file.param_rec import args


class NCF(nn.Module):
    def __init__(self, user_num, item_num, pretrain_features):
        super(NCF, self).__init__()
        self.dropout = args.dropout
        self.item_num = item_num
        self.embed_user_MLP = nn.Embedding(user_num, args.dim)
        self.input_dim = args.dim * 2
        self._init_item_em(pretrain_features)
        MLP_modules = []
        for i in range(args.num_layers):
            input_size = int(self.input_dim / (2 ** i))
            MLP_modules.append(nn.Dropout(p=self.dropout))
            MLP_modules.append(nn.Linear(input_size, input_size // 2))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        self.predict_layer = nn.Linear(int(self.input_dim / (2 ** args.num_layers)), 1)

        self._init_weight_()

    def _init_weight_(self):
        """ We leave the weights initialization here. """
        nn.init.normal_(self.embed_user_MLP.weight, std=0.01)

        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_uniform_(self.predict_layer.weight,
                                 a=1, nonlinearity='sigmoid')

    def _init_item_em(self, pretrain_features):
        if args.pretrain:
            self.feat_item = nn.Embedding(self.item_num, pretrain_features.shape[-1])
            self.feat_item.weight.data.copy_(torch.from_numpy(pretrain_features))
        else:
            self.embed_item_init = nn.Embedding(self.item_num, args.dim)
            nn.init.normal_(self.embed_item_init.weight, std=0.01)

    def get_item_em(self, item_id):
        if args.pretrain:
            item_em = self.feat_item(item_id)
        else:
            item_em = self.embed_item_init(item_id)
        return item_em

    def forward(self, user, pos_item, neg_item):
        embed_user_MLP = self.embed_user_MLP(user)
        embed_item_pos_MLP = self.get_item_em(pos_item)
        embed_item_neg_MLP = self.get_item_em(neg_item)

        interaction_pos = torch.cat((embed_user_MLP, embed_item_pos_MLP), -1)
        output_pos_MLP = self.MLP_layers(interaction_pos)
        prediction_pos = self.predict_layer(output_pos_MLP).view(-1)

        interaction_neg = torch.cat((embed_user_MLP, embed_item_neg_MLP), -1)
        output_neg_MLP = self.MLP_layers(interaction_neg)
        prediction_neg = self.predict_layer(output_neg_MLP).view(-1)
        loss_value = - \
            torch.sum(torch.log2(torch.sigmoid(prediction_pos - prediction_neg)))
        return prediction_pos, prediction_neg, loss_value

    def loss(self, users, pos_items, neg_items):
        pos_scores, neg_scores = self.forward(users, pos_items, neg_items)
        loss_value = - \
           torch.sum(torch.log2(torch.sigmoid(pos_scores - neg_scores)))
        return loss_value
