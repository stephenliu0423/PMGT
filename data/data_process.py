# Import Modules

import gzip
import json
import os
from collections import Counter
from copy import deepcopy
from datetime import datetime
from functools import partial

import backoff
import joblib
import networkx as nx
import numpy as np
import pandas as pd
import requests
import scipy.sparse as sp
import timm

from joblib import Parallel, delayed

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer
import pickle
import random
import os
from collections import OrderedDict
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
from PIL import Image
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
os.environ['CUDA_VISIBLE_DEVICES'] = "1"



class AmazonReviewImageDataset(torch.utils.data.Dataset):
    def __init__(
        self, root: str, transforms: Callable, item_ids: Optional[np.ndarray] = None
    ) -> None:
        self.root = root
        self.transforms = transforms
        self.item_ids = item_ids
        self.images, self._num_images = self._get_image_list()

    @property
    def num_images(self) -> Dict[str, int]:
        return self._num_images

    def _get_image_list(self) -> Tuple[List[str], Dict[str, int]]:
        images = []
        num_images = OrderedDict()
        for item_id in os.listdir(self.root):
            if self.item_ids is not None and item_id not in self.item_ids:
                continue
            imagefile_list = os.listdir(os.path.join(self.root, item_id))
            num_images[item_id] = len(imagefile_list)
            for image_name in imagefile_list:
                images.append(os.path.join(item_id, image_name))
        return images, num_images

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(os.path.join(self.root, self.images[idx])).convert("RGB")
        return self.transforms(img)

    def __len__(self) -> int:
        return len(self.images)


class AmazonReviewTextDataset(torch.utils.data.Dataset):
    def __init__(self, texts: Dict[str, np.ndarray]) -> None:
        self.texts, self._num_texts = self._get_text_list(texts)

    @property
    def num_texts(self) -> Dict[str, int]:
        return self._num_texts

    def _get_text_list(
        self, item_texts: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        texts = []
        num_texts = OrderedDict()
        for item_id, text_array in item_texts.items():
            num_texts[item_id] = len(text_array)
            texts.append(text_array)
        return np.concatenate(texts), num_texts

    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]

    def __len__(self) -> int:
        return len(self.texts)


def text_collate_fn(
    batch: Iterable[str],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 128,
) -> Dict[str, torch.Tensor]:
    return tokenizer(
        batch,
        max_length=max_length,
        padding="max_length",
        truncation="longest_first",
        return_tensors="pt",
    )


def _giveup(e):
    return str(e) == "404"

@backoff.on_exception(
    backoff.expo,
    (requests.exceptions.RequestException, requests.exceptions.ConnectionError),
    max_time=30,
    max_tries=5,
    giveup=_giveup,
)

def download_image(filepath, image_url):
    if os.path.exists(filepath):
        return False

    try:
        r = requests.get(image_url, stream=True)
    except requests.exceptions.MissingSchema:
        return False

    if r.status_code == 404:
        return False
    elif r.status_code != 200:
        raise requests.exceptions.RequestException(r.status_code)

    with open(filepath, "wb") as f:
        for chunk in r.iter_content(1024):
            f.write(chunk)

    return True


def get_feat_init_emb(node_size, items, feats, item_to_idx):
    feat_init_emb = np.empty((node_size, feats.shape[1]), dtype=np.float32)

    for i, item in enumerate(items):
        if item not in item_to_idx:
            feat_init_emb[i] = np.random.normal(size=feats.shape[1])
        else:
            feat_init_emb[i] = feats[item_to_idx[item]]

    return feat_init_emb


def save_ncf_data(data_dir, down_df, G):
    # save downstream NCF dataset

    rec_df = down_df[down_df['asin'].isin(G.nodes.keys())]

    node_encoder = joblib.load(os.path.join(data_dir, "node_encoder"))
    user_encoder = LabelEncoder().fit(rec_df['reviewerID'].unique())
    item_to_idx = {v: i for i, v in enumerate(node_encoder.classes_)}
    user_to_idx = {v: i for i, v in enumerate(user_encoder.classes_)}

    random_state = np.random.RandomState(2022)
    train_df, test_df = train_test_split(rec_df, test_size=0.2, random_state=random_state)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    print("overall interaction", len(rec_df))
    print("train, test interaction", len(train_df), len(test_df))
    print("user, item rec dataset", len(rec_df['reviewerID'].unique()), len(rec_df['asin'].unique()))

    train_per_user = train_df.groupby(by="reviewerID").apply(lambda r: r["asin"].tolist())
    tr_user_ids = train_df['reviewerID'].unique()
    user_train_items = dict()
    tr_n = 0
    for user in tqdm(tr_user_ids):
        user_id = user_to_idx[user]

        item_ids = [item_to_idx[i] for i in train_per_user[user]]
        user_train_items[user_id] = set(item_ids)
        tr_n += len(item_ids)

    with open(data_dir + 'downstream/train_rec.pickle', 'wb') as f:
        pickle.dump(user_train_items, f)
    test_per_user = test_df.groupby(by="reviewerID").apply(lambda r: r["asin"].tolist())
    te_user_ids = test_df['reviewerID'].unique()
    user_test_items = dict()
    te_n = 0
    for user in tqdm(te_user_ids):
        user_id = user_to_idx[user]

        item_ids = [item_to_idx[i] for i in test_per_user[user]]
        user_test_items[user_id] = set(item_ids)
        te_n += len(item_ids)
    with open(data_dir + 'downstream/test_rec.pickle', 'wb') as f:
        pickle.dump(user_test_items, f)

    print("train/test set number", tr_n, te_n)


def save_dcn_data(data_dir, down_df, G):

    rec_df = down_df[down_df['asin'].isin(G.nodes.keys())]

    node_encoder = joblib.load(os.path.join(data_dir, "node_encoder"))
    user_encoder = LabelEncoder().fit(rec_df['reviewerID'].unique())
    item_to_idx = {v: i for i, v in enumerate(node_encoder.classes_)}
    user_to_idx = {v: i for i, v in enumerate(user_encoder.classes_)}

    items_per_user = rec_df.groupby(by="reviewerID").apply(lambda r: r["asin"].tolist())
    items_set = list(rec_df['asin'].unique())
    users_set = rec_df['reviewerID'].unique()
    dcn_data = []
    for user in tqdm(users_set):
        user_id = user_to_idx[user]
        item_ids = [item_to_idx[i] for i in items_per_user[user]]
        items_user = set(item_ids)
        for pos_iid in item_ids:
            dcn_data.append([int(user_id), int(pos_iid), 1])
            for i in range(5):
                neg_iid = random.randint(0, len(items_set) - 1)
                while neg_iid in items_user:
                    neg_iid = random.randint(0, len(items_set))
                dcn_data.append([int(user_id), int(neg_iid), 0])

    dcn_df = pd.DataFrame(dcn_data, columns=['u', 'i', 'label'])

    random_state = np.random.RandomState(2022)
    train_dcn, test_dcn = train_test_split(dcn_df, test_size=0.2, random_state=random_state)
    tr_dcn_list = train_dcn.values.tolist()
    te_dcn_list = test_dcn.values.tolist()
    with open(data_dir + 'downstream/train_ctr.pickle', 'wb') as f:
        pickle.dump(np.array(tr_dcn_list), f)
    with open(data_dir + 'downstream/test_ctr.pickle', 'wb') as f:
        pickle.dump(np.array(te_dcn_list), f)
    print(len(tr_dcn_list), len(te_dcn_list))


def save_graph_edge(data_dir, G, graph_data):
    edge_path = data_dir + 'edge_org.txt'

    file_handle = open(edge_path, 'w')
    file_handle.write('src_id:int64\tdst_id:int64\tweight:float\tfeature:string')

    node_encoder = joblib.load(os.path.join(data_dir, "node_encoder"))
    node_to_idx = {v: i for i, v in enumerate(node_encoder.classes_)}

    for trupe in graph_data:
        file_handle.write('\n')
        src, dst, w = trupe
        w = (np.log(w) + 1) / (np.log(np.sqrt(G.degree[src] * G.degree[dst])) + 1)
        src_idx = node_to_idx[src]
        dst_idx = node_to_idx[dst]
        feat_ = 0
        file_handle.write(str(src_idx) + '\t' + str(dst_idx) +
                          '\t' + str(w) + '\t' + str(feat_))
    file_handle.close()


def save_graph_node(data_dir, G, image_root_path, download_list, meta_df, review_text):
    node_encoder = LabelEncoder().fit(list(G.nodes.keys()))
    joblib.dump(node_encoder, os.path.join(data_dir, "node_encoder"))

    node_encoder = joblib.load(os.path.join(data_dir, "node_encoder"))

    visual_feats, visual_feats_mapping = visual_feature(image_root_path=image_root_path, download_list=download_list, meta_df=meta_df)

    textual_feats, textual_feats_mapping = textual_feature(review_text=review_text)

    node_size = len(node_encoder.classes_)
    item_to_idx = {item: i for i, item in enumerate(visual_feats_mapping)}
    visual_init_emb = get_feat_init_emb(
        node_size, node_encoder.classes_, visual_feats, item_to_idx
    )

    item_to_idx = {item: i for i, item in enumerate(textual_feats_mapping)}
    textual_init_emb = get_feat_init_emb(
        node_size, node_encoder.classes_, textual_feats, item_to_idx
    )
    mm_feat_init_emb = np.concatenate([visual_init_emb, textual_init_emb], axis=1)
    print("multi modal feature shape")
    print(visual_init_emb.shape)
    print(textual_init_emb.shape)
    print(mm_feat_init_emb.shape)

    text_path = data_dir + 'node_feat.txt'
    file_handle = open(text_path, 'w')
    file_handle.write('id:int64\tfeature:string')
    for i, feat in enumerate(mm_feat_init_emb):
        file_handle.write('\n')
        file_handle.write(str(i) + '\t' + ','.join(str(i) for i in feat))
    file_handle.close()


def save_tr_te_node(data_dir, node_num):
    # save train\test node feature
    train_path = data_dir + 'node_feat_train.txt'
    test_path = data_dir + 'node_feat_test.txt'
    train_handle = open(train_path, 'w')
    train_handle.write('id:int64\tfeature:string')
    test_handle = open(test_path, 'w')
    test_handle.write('id:int64\tfeature:string')

    text_path = data_dir + 'node_feat.txt'
    select_num = int(node_num/10)
    select = np.random.choice(list(range(node_num)), select_num, replace=False)
    print(len(select))
    with open(text_path, 'r') as f:
        for feat in f.readlines():
            if "int64" in feat:
                continue

            feat = feat.strip('\n')
            id_feature = feat.split('\t')
            iid = id_feature[0]
            feature = id_feature[1:]
            if int(iid) in select:
                test_handle.write('\n')
                test_handle.write(str(iid) + '\t' + ','.join(feature))
            else:
                train_handle.write('\n')
                train_handle.write(str(iid) + '\t' + ','.join(feature))

    train_handle.close()
    test_handle.close()


def visual_feature(image_root_path, download_list, meta_df):
    results = Parallel(n_jobs=50, prefer="threads")(
        delayed(download_image)(f, u) for f, u in tqdm(download_list)
    )
    print("down image over")

    model = timm.create_model("inception_v4", pretrained=True)
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    dataset = AmazonReviewImageDataset(
        image_root_path, transforms=transform, item_ids=meta_df["asin"].unique()
    )

    dataloader = DataLoader(dataset, batch_size=32, num_workers=8)

    model.cuda()
    model.eval()

    visual_feats = []
    for batch_x in tqdm(dataloader, total=len(dataloader)):
        batch_x = batch_x.cuda()
        with torch.no_grad():
            feat = model.global_pool(model.forward_features(batch_x))
            visual_feats.append(feat.cpu())

    visual_feats = torch.cat(visual_feats)
    print(visual_feats.size())

    item_visual_feats = []
    start = 0
    for num in tqdm(dataset.num_images.values()):
        end = start + num
        item_visual_feats.append(visual_feats[start:end].mean(dim=0))
        start = end
    item_visual_feats = torch.stack(item_visual_feats).numpy()
    item_mapping = np.array([item_id for item_id in dataset.num_images.keys()])
    return item_visual_feats, item_mapping


def textual_feature(review_text):
    dataset = AmazonReviewTextDataset(review_text)
    model_name = "bert-base-uncased"

    print("total description number", len(dataset.texts))

    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model.cuda()
    model.eval()

    dataloader = DataLoader(
        dataset,
        batch_size=64,
        num_workers=16,
        collate_fn=partial(text_collate_fn, tokenizer=tokenizer),
    )

    text_feats = []

    for batch_x in tqdm(dataloader, total=len(dataloader)):
        batch_x = {k: v.cuda() for k, v in batch_x.items()}
        with torch.no_grad():
            text_feats.append(model(**batch_x)[0][:, 0].cpu())

    text_feats = torch.cat(text_feats)

    item_textual_feats_list = []
    start = 0
    for num in tqdm(dataset.num_texts.values()):
        end = start + num
        item_textual_feats_list.append(text_feats[start:end].mean(dim=0))
        start = end
    item_textual_feats = torch.stack(item_textual_feats_list).numpy()
    item_mapping = np.array([item_id for item_id in dataset.num_texts.keys()])
    return item_textual_feats, item_mapping


def build_graph(graph_df, threshold):
    users_per_item = graph_df.groupby(by="asin").apply(lambda r: r["reviewerID"].tolist())

    item_ids = graph_df["asin"].unique()

    item_encoder = LabelEncoder().fit(item_ids)
    user_encoder = LabelEncoder().fit(graph_df["reviewerID"].unique())

    item_user_mat = sp.dok_matrix(
        (len(item_encoder.classes_), len(user_encoder.classes_)), dtype=np.int32
    )

    item_to_idx = {v: i for i, v in enumerate(item_encoder.classes_)}
    user_to_idx = {v: i for i, v in enumerate(user_encoder.classes_)}

    idx = 0
    for item in tqdm(item_ids):
        item_id = item_to_idx[item]
        idx += len(users_per_item[item])
        user_ids = [user_to_idx[u] for u in users_per_item[item]]
        item_user_mat[item_id, user_ids] = 1

    item_user_mat_csr = item_user_mat.tocsr()
    item_item_mat = item_user_mat_csr @ item_user_mat_csr.T
    ## example for @
    ## 1，0，0，1      1，0，0      2, 1, 1
    ## 0，1，0，1      0，1，0      1, 2, 1
    ## 0，0，0，1      0，0，0      1, 1, 1
    ##                1，1，1

    item_item_mat.setdiag(0)
    item_item_mat.eliminate_zeros()

    graph_data = []
    for i, row in enumerate(tqdm(item_item_mat, total=item_item_mat.shape[0])):
        for j, r in zip(row.indices, row.data):
            if r >= threshold:
                graph_data.append((item_encoder.classes_[i], item_encoder.classes_[j], r))
    print("edge number of graph", len(graph_data))
    G = nx.Graph()
    G.add_weighted_edges_from(graph_data)

    for u, v, w in tqdm(G.edges.data("weight")):
        w = (np.log(w) + 1) / (np.log(np.sqrt(G.degree[u] * G.degree[v])) + 1)
        G.edges[u, v]["weight"] = w

    print("node num", G.number_of_nodes())
    print("edge num", 2 * G.number_of_edges())
    return G, graph_data

def load_raw_data(data_dir, file_name):
    with gzip.open(os.path.join(data_dir, file_name)) as f:
        data = [json.loads(l.strip()) for l in tqdm(f)]

    df = pd.DataFrame.from_dict(data)
    df['reviewDateTime'] = df['unixReviewTime'].map(lambda x: datetime.fromtimestamp(x))
    df = df.sort_values(by='reviewDateTime')
    print("raw data length", len(df))

    criterion = datetime(2015, 1, 1)
    graph_df = df[df['reviewDateTime'] <= criterion].reset_index(drop=True)
    down_df = df[df['reviewDateTime'] > criterion].reset_index(drop=True)
    print("graph df length", len(graph_df))
    print("down df length", len(down_df))
    return graph_df, down_df

def load_meta_data(data_dir, meta_name, G):
    with gzip.open(os.path.join(data_dir, meta_name)) as f:
        meta_data = [json.loads(l.strip()) for l in tqdm(f)]

    meta_df = pd.DataFrame.from_dict(meta_data)

    g_node = set(G.nodes)
    print("graph node number", len(g_node))

    meta_df = meta_df[meta_df['asin'].isin(g_node)]

    print("before drop duplicates", len(meta_df))
    meta_df = meta_df.drop_duplicates('asin', keep='last')
    print("after drop duplicates", len(meta_df))

    image_root_path = os.path.join(data_dir, "images")
    os.makedirs(image_root_path, exist_ok=True)
    download_list = []
    counter = Counter()
    it = []
    for index, row in meta_df[~pd.isna(meta_df["imageURL"])].iterrows():
        for i, image_url in enumerate(row["imageURL"]):
            ext = os.path.splitext(image_url)[1]
            item_id = row["asin"]
            filepath = os.path.join(image_root_path, item_id, f"{counter[item_id]}{ext}")
            counter[item_id] += 1
            download_list.append((filepath, image_url))

            if not os.path.exists(os.path.dirname(filepath)):
                os.makedirs(os.path.dirname(filepath), exist_ok=True)

            it.append(item_id)
    print(len(set(it)))
    print(len(download_list))
    print(len(meta_df["asin"].unique()))
    print(len(next(os.walk(image_root_path))[1]))
    print("read url over")

    review_text = (
        meta_df[~pd.isna(meta_df["description"])]
            .groupby("asin")
            .apply(lambda r: r["description"].values[0])
    )
    review_text = review_text.to_dict()
    print("asin number with description", len(review_text))

    old_review_text = deepcopy(review_text)
    for key in old_review_text.keys():
        values = old_review_text[key]
        if len(values) == 0:
            del review_text[key]

    print("asin number with description after del empty", len(review_text))

    return image_root_path, download_list, meta_df, review_text

if __name__ == '__main__':
    # pd.set_option('display.max_columns', None)

    # Data Preprocessing
    # [Amazon Review Datasets](https://nijianmo.github.io/amazon/index.html)
    # VG Dataset
    # - Video_Games_5.json.gz
    # - meta_Video_Games.json.gz

    data_dir = '../data/'
    pre_pocess_dir = '../data/video/'

    file_name = 'Video_Games_5.json.gz'
    meta_name = 'meta_Video_Games.json.gz'

    # threshold = 3 for video and tools, threshold = 4 for toys
    # threshold = 3 on toys dataset will bring NAN error due to bert optimizer
    # caused by (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
    # code seems no error, maybe influenced by the different vision of environment
    
    threshold = 3

    graph_df, down_df = load_raw_data(data_dir, file_name)
    G, graph_data = build_graph(graph_df, threshold)
    image_root_path, download_list, meta_df, review_text = load_meta_data(data_dir, meta_name, G)
    save_graph_node(pre_pocess_dir, G, image_root_path, download_list, meta_df, review_text)
    save_graph_edge(pre_pocess_dir, G, graph_data)
    save_tr_te_node(pre_pocess_dir, G.number_of_nodes())
    save_ncf_data(pre_pocess_dir, down_df, G)
    save_dcn_data(pre_pocess_dir, down_df, G)

