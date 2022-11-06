import pickle
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

FRAME_FEAT_DIM = 1536


def read_local_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def save_local_pkl(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def read_local_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = f.read()
    return data


def to_tensor(data, use_gpu=False, is_long=True):
    if is_long:
        data = torch.LongTensor(data)
    else:
        data = torch.FloatTensor(data.float())
    if use_gpu:
        data = data.cuda()
    return data


def early_stopping(log_value, best_value, stopping_step, flag_step=3):
    # early stopping strategy:
    if best_value is None or log_value > best_value:
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1
    if stopping_step >= flag_step:
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop


def normalization(data):
    mm = MinMaxScaler()
    data_mm = mm.fit_transform(data)
    return data_mm


def standardization(data):
    ss = StandardScaler()
    data_ss = ss.fit_transform(data)
    return data_ss


