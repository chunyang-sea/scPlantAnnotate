# -*- coding: utf-8 -*-
import os
import gc
import argparse
import json
import random
import math
import random
from functools import reduce
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split, ShuffleSplit, StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support, classification_report
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CyclicLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
#import torch.distributed as dist

from performer import PlantLM
import scanpy as sc
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=0, help='Local process rank.')
parser.add_argument("--bin_num", type=int, default=5, help='Number of bins.')
parser.add_argument("--gene_num", type=int, default=20000, help='Number of genes.')
parser.add_argument("--embed_dim", type=int, default=150, help='embedding dimension.')
parser.add_argument("--heads", type=int, default=10, help='number of heads in Performer.')
parser.add_argument("--num_layer", type=int, default=5, help='Number of transformer layers.')
parser.add_argument("--h_dim", type=int, default=128, help='hidden dimension in MLP.')
parser.add_argument("--batch_size", type=int, default=1, help='Number of batch size.')
parser.add_argument("--data_path", type=str, default='', help='Path of data for evaluation.')
parser.add_argument("--dataset", type=str, default='', help='dataset for evaluation.')
parser.add_argument("--model_path", type=str, default='', help='Path of pretrained model.')
parser.add_argument("--gene_embed", action="store_true", help='Using gene embedding or not.')
parser.add_argument("--condition_embed", action="store_true", help='Using condition embedding or not.')
parser.add_argument("--condition_column", type=str, default='Dataset', help='adata.obs column name for condition embedding.')
parser.add_argument("--conditiontype_path", type=str, default='', help='condition type labels path')
parser.add_argument("--mlp_embedding", action='store_true', help='use mlp embedding.')
parser.add_argument("--log_file", type=str, default='log.txt', help='log file path')
parser.add_argument("--output_dir", type=str, default='.', help='output directory for results')
parser.add_argument("--prediction_file", type=str, default='pred.csv', help='cell type prediction file path')
parser.add_argument("--score_file", type=str, default='score.csv', help='score file path')
parser.add_argument("--cm_file", type=str, default='cm.csv', help='confusion matrix file path')

args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
print("Starting evaluation")
local_rank = args.local_rank
CLASS = args.bin_num + 2
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

class SCDataset(Dataset):
    def __init__(self, data, labels, meta, cell_ids, binning_expression=True):
        super().__init__()
        self.data = data
        self.labels = torch.from_numpy(labels)
        self.meta = torch.from_numpy(meta) if meta is not None else None
        self.cell_ids = cell_ids
        self.binning_expression = binning_expression

    def __getitem__(self, index):
        row = self.data[index]
        if hasattr(row, 'toarray'):
            full_seq = row.toarray()[0]
        else:
            full_seq = row
        if self.binning_expression:
            full_seq[full_seq > (CLASS - 2)] = CLASS - 2
            full_seq = torch.from_numpy(full_seq).long()
            full_seq = torch.cat((full_seq, torch.tensor([0])))
        else:
            full_seq = torch.from_numpy(full_seq).float()
        if self.meta is not None:
            return {
                'x': full_seq,
                'y': self.labels[index],
                'meta': self.meta[index],
                'cell_ids': self.cell_ids[index]
            }
        else:
            return {
                'x': full_seq,
                'y': self.labels[index],
                'cell_ids': self.cell_ids[index]
            }

    def __len__(self):
        return self.data.shape[0]

class MLPClassifier(nn.Module):
    def __init__(self, seq_len, embed_dim, h_dim, out_dim, dropout=0.):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, (1, embed_dim))
        self.fc1 = nn.Linear(seq_len, 512)
        self.fc2 = nn.Linear(512, h_dim)
        self.fc3 = nn.Linear(h_dim, out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, embed_dim)
        x = x.unsqueeze(1) 
        x = self.relu(self.conv1(x))
        x = x.view(x.size(0), -1)       # flatten to (batch, seq_len)

        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)

        return x

adata = sc.read_h5ad(args.data_path)

if args.dataset == '':
    print('dataset is empty, use all data')
    print(f"shape: {adata.X.shape}")
else:
    data_list = args.dataset.split(",")
    data = adata[adata.obs['Dataset'] == data_list[0], :]
    if local_rank == 0:
        print(f'add {data_list[0]}, shape: {data.X.shape}')
    for ds in data_list[1:]:
        data = data.concatenate(adata[adata.obs['Dataset'] == ds, :])
        if local_rank == 0:
            print(f'after adding {ds}, shape: {data.X.shape}')
    adata = data

path = args.model_path
ckpt = torch.load(path, map_location='cpu', weights_only=False)

if 'output_node_names' in ckpt:
    label_dict_list = ckpt['output_node_names']
else:
    assert False, f"did not find output_node_names in {args.model_path}"

reverse_dict = {ctype: idx for idx, ctype in enumerate(label_dict_list)}
adata.obs['CelltypeID'] = adata.obs['Celltype'].map(reverse_dict)
if adata.obs['CelltypeID'].isnull().any():
    assert False, "data has cell types that can't find mapping in input model's output nodes"
else:
    adata.obs['CelltypeID'] = adata.obs['CelltypeID'].astype(int)

label_list = list(range(len(label_dict_list)))

if args.condition_embed:
    condition_array = np.loadtxt(args.conditiontype_path, dtype=str, delimiter=",")
    print("Loaded conditions:", condition_array)
    condition_to_index = {cond: idx for idx, cond in enumerate(condition_array)}
    test_conditions = adata.obs[args.condition_column].astype(str).values
    conditions = np.array([condition_to_index[c] for c in test_conditions])
else:
    conditions = None

data = adata.X
label = adata.obs['CelltypeID'].to_numpy()
cell_names = adata.obs.index.tolist()

acc = []
f1 = []
f1w = []
test_dataset = SCDataset(data, label, conditions, cell_names, not args.mlp_embedding)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
print(f"Number of samples in val_dataset: {len(test_dataset)}")

model = PlantLM(
    num_tokens = CLASS,
    dim = args.embed_dim,
    depth = args.num_layer,
    max_seq_len = args.gene_num + 1,
    heads = args.heads,
    local_attn_heads = 0,
    mlp_embedding=args.mlp_embedding,
    gene_emb = args.gene_embed,
    condition_emb = args.condition_embed,
    num_conditions = len(condition_array) if args.condition_embed else 0,
)
nclass = len(label_dict_list)
model.to_out = MLPClassifier(args.gene_num + 1, args.embed_dim, args.h_dim, nclass)

model.load_state_dict(ckpt['model_state_dict'])
model = model.to(device)
loss_fn = nn.CrossEntropyLoss(weight=None).to(device)
softmax = nn.Softmax(dim=-1)

model.eval()
running_loss = 0.0
predictions = []
truths = []
nskip = 0
cells = []
pred_ctypes = []
with torch.no_grad():
    for index, batch in enumerate(test_loader):
        index += 1
        print(f"\rProcessing batch {index}/{len(test_loader)}", end='', flush=True)
        if index == 1000:
            break

        data_v = batch['x'].to(device)
        labels_v = batch['y'].to(device)
        meta_v = batch['meta'].to(device) if 'meta' in batch else None
        names_v = batch['cell_ids']
        logits = model(data_v) if meta_v is None else model.forward_with_meta(data_v, meta_v)

        if labels_v.item() < 0 or labels_v.item() >= nclass:
            print(f'index={index} label={labels_v.item()} is out of bound, skip this sample')
            nskip += 1
            continue
        logits = model(data_v)
        loss = loss_fn(logits, labels_v)
        running_loss += loss.item()
        final_prob = softmax(logits)
        final = final_prob.argmax(dim=-1)
        final[np.amax(np.array(final_prob.cpu()), axis=-1) < 0.0] = -1
        predictions.append(final)
        truths.append(labels_v)
        cells.extend(names_v)
        pred_ctypes.extend([label_dict_list[i] for i in final])
    del data_v, labels_v, names_v, logits, final_prob, final
    # gather
    no_drop = predictions != -1
    if no_drop:
        predictions = np.array([tensor.cpu().detach().numpy() for tensor in predictions])
        truths = np.array([tensor.cpu().detach().numpy() for tensor in truths])
    else:
        predictions = np.array((predictions[no_drop]).cpu())
        truths = np.array((truths[no_drop]).cpu())
    cur_acc = accuracy_score(truths, predictions)
    f1 = f1_score(truths, predictions, average='macro')
    val_loss = running_loss / (index - nskip)
    print(f'number of skipped samples: {nskip}')
    print(f'    ==  Accuracy: {cur_acc:.6f} | F1 Score: {f1:.6f}  ==')
    print(confusion_matrix(truths, predictions))
    print(classification_report(truths, predictions, labels=label_list, target_names=label_dict_list, digits=4))

    truths = truths.ravel() # flatten the array from shape (n, 1) to (n,)
    predictions = predictions.ravel()
    true_labels = np.array([label_dict_list[i] for i in truths])
    pred_labels = np.array([label_dict_list[i] for i in predictions])
    clf_report = pd.DataFrame(classification_report(true_labels, pred_labels, output_dict=True)).T
    clf_report.to_csv(f'{args.output_dir}/{args.score_file}', index=True, header=True, sep='\t')
    print(clf_report)

    labels = np.unique(np.concatenate([true_labels, pred_labels]))
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)

    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    df_cm.to_csv(f"{args.output_dir}/{args.cm_file}")

# Create a DataFrame with cell names and predictions
df = pd.DataFrame({
    'Cell_Name': cells,
    'Prediction': pred_ctypes
})
df.to_csv(f'{args.output_dir}/{args.prediction_file}', index=False, header=False)

print("Finished Evaluation")