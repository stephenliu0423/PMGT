import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import torch
from time import time
import PMGT_v2.utils.Metrics as metrics
from PMGT_v2.utils.utils import early_stopping, to_tensor
from PMGT_v2.down_rec.data_loader import RecData
from PMGT_v2.config_file.param_rec import args
from PMGT_v2.config_file.config_downtream import get_rec_config

from PMGT_v2.down_rec.NCF import NCF


class Net(object):
    def __init__(self):
        super().__init__()
        self.data = RecData()
        config = get_rec_config(args.data_type)
        self.model = NCF(config['num_users'], config['num_items'], self.data.pretrain_features)
        if args.useGPU:
            self.model.cuda()
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                    weight_decay=args.l2_re)

    def train_model(self):
        print('################Training model###################')
        stopping_step = 0
        best_cu = None
        Ks = eval(args.Ks)
        for epoch in range(1, args.epochs + 1):
            t1 = time()
            train_loss = self.train()
            t2 = time()
            print('epoch{: d}: train_time:{: .2f}s, train_loss:{: .4f}'.format(
                epoch, t2 - t1, train_loss))
            if epoch % 5 == 0:
                print('Testing start...')
                precision, recall, ndcg, auc = self.accuracy(Ks)
                best_cu, stopping_step, should_stop = early_stopping(
                    recall[1], best_cu, stopping_step, flag_step=5)
                if recall[1] == best_cu and args.rep_flag:
                    perf_str = "precision=[%s],\nrecall=[%s],\nndcg=[%s],\nauc=[%.4f]" % \
                        (' '.join(['%.4f' % r for r in precision]),
                         ' '.join(['%.4f' % r for r in recall]),
                         ' '.join(['%.4f' % r for r in ndcg]),
                         auc)
                    print(perf_str)
                if should_stop:
                    break

    def train(self):
        self.model.train()
        epoch_loss = 0
        train_loader = DataLoader(dataset=self.data, batch_size=args.batch_size, shuffle=True)
        batch_num = 0
        for uid, iid_pos, iid_neg in train_loader:
            self.optimizer.zero_grad()
            uid = uid.long()
            iid_pos = iid_pos.long()
            iid_neg = iid_neg.long()
            if args.useGPU:
                uid = uid.cuda()
                iid_pos = iid_pos.cuda()
                iid_neg = iid_neg.cuda()
            _, _, loss = self.model(uid, iid_pos, iid_neg)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            batch_num += 1
        return epoch_loss / batch_num

    def accuracy(self, Ks):
        self.model.eval()
        sum_user = 0
        precision = [0 for k in Ks]
        recall = [0 for k in Ks]
        ndcg = [0 for k in Ks]
        batch_size = args.batch_size
        auc_sum = 0
        for user, pos_items, neg_items in self.data.read_test_data():
            sum_user += 1
            all_items = np.hstack((pos_items, neg_items))
            all_scores = np.zeros(len(all_items))
            num_pos = len(pos_items)
            num = int(len(all_items) / (2 * batch_size))
            s3 = 0
            for i in range(0, num):
                s1 = 2 * i * batch_size
                s2 = (2 * i + 1) * batch_size
                s3 = (2 * i + 2) * batch_size
                batch_user_tensor = to_tensor([user] * batch_size, args.useGPU)
                batch_pos_tensor = to_tensor(all_items[s1:s2], args.useGPU)
                batch_neg_tensor = to_tensor(all_items[s2:s3], args.useGPU)
                with torch.no_grad():
                    score1, score2, _ = self.model(
                        batch_user_tensor, batch_pos_tensor, batch_neg_tensor)
                all_scores[s1:s2] = score1.cpu().detach().numpy()
                all_scores[s2:s3] = score2.cpu().detach().numpy()
            if len(all_items[s3:]) != 0:
                batch_user_tensor = to_tensor([user] * len(all_items[s3:]), args.useGPU)
                batch_pos_tensor = to_tensor(all_items[s3:], args.useGPU)
                batch_neg_tensor = to_tensor(all_items[s3:], args.useGPU)
                with torch.no_grad():
                    score1, _, _ = self.model(
                        batch_user_tensor, batch_pos_tensor, batch_neg_tensor)
                all_scores[s3:] = score1.cpu().detach().numpy()
            r, auc = metrics.ranklist_by_sort(
                pos_items, all_items, -all_scores)
            auc_sum += auc
            for i, k in enumerate(Ks):
                precision[i] += metrics.precision_at_k(r, k)
                recall[i] += metrics.recall_at_k(r, k, num_pos)
                ndcg[i] += metrics.ndcg_at_k(r, k)
        return np.array(precision) / sum_user, np.array(recall) / sum_user, np.array(ndcg) / sum_user, auc_sum / sum_user


if __name__ == '__main__':
    rec_task = Net()
    rec_task.train_model()
