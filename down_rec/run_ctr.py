from PMGT_v2.down_rec.DCN_cross import CDNet
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
from time import time
from sklearn.metrics import roc_auc_score
from PMGT_v2.utils.utils import early_stopping
from PMGT_v2.down_rec.data_loader_cross import RecData, read_features_table
from PMGT_v2.config_file.param_rec import args
from PMGT_v2.config_file.config_downtream import get_rec_config


class Net(object):
    def __init__(self):
        super().__init__()
        features = read_features_table()
        self.criterion = torch.nn.BCELoss()
        config = get_rec_config(args.data_type)
        self.model = CDNet(config['num_users'], config['num_items'], features)
        if args.useGPU:
            self.model.cuda()
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                    weight_decay=args.l2_re)

    def train_model(self):
        print('################Training model###################')
        stopping_step = 0
        best_cu = None
        for epoch in range(1, args.epochs + 1):
            t1 = time()
            train_loss = self.train()
            t2 = time()
            print('epoch{: d}: train_time:{: .2f}s, train_loss:{: .4f}'.format(
                epoch, t2 - t1, train_loss))
            if epoch % 5 == 0:
                print('Testing start...')
                auc = self.accuracy()
                best_cu, stopping_step, should_stop = early_stopping(
                    auc, best_cu, stopping_step, flag_step=5)
                if auc == best_cu and args.rep_flag:
                    print('auc=[%.4f]' % auc)

    def train(self):
        self.model.train()
        epoch_loss = 0
        train_data = RecData('train_ctr.pickle')
        train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size)
        for uid, iid, label in train_loader:
            self.optimizer.zero_grad()
            uid = uid.long()
            iid = iid.long()
            label = label.float()
            if args.useGPU:
                uid = uid.cuda()
                iid = iid.cuda()
                label = label.cuda()
            scores = self.model(uid, iid)
            loss = self.criterion(scores, label)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / len(train_loader)

    def accuracy(self):
        self.model.eval()
        test_data = RecData('test_ctr.pickle')
        test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size)
        all_label = []
        all_scores = []
        for uid, iid, label in test_loader:
            uid = uid.long()
            iid = iid.long()
            label = label.float()
            if args.useGPU:
                uid = uid.cuda()
                iid = iid.cuda()
                label = label.cuda()
            with torch.no_grad():
                scores = self.model(uid, iid)
            all_label.extend(list(label.cpu().detach().numpy()))
            all_scores.extend(list(scores.cpu().detach().numpy()))
        auc = roc_auc_score(y_true=all_label, y_score=all_scores)
        return auc


if __name__ == '__main__':
    rec_task = Net()
    rec_task.train_model()
