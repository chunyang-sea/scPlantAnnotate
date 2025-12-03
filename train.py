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
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
os.environ["NCCL_DEBUG"] = "OFF"  # or :"WARN" / "OFF" / "ERROR"

import torch
from torch import nn
from torch.optim import Adam, SGD, AdamW
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CyclicLR
from torch.utils.data import DataLoader, Dataset
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from contextlib import nullcontext

from performer import PlantLM
import scanpy as sc
import anndata as ad
from utils import *
import pickle as pkl

BIN_NUM = 5
CLASS = BIN_NUM + 2

from torch.optim.lr_scheduler import LambdaLR
def linear_warmup_decay(warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps))
        )
    return lr_lambda

class SCDataset(Dataset):
    def __init__(self, device, data, labels, meta, sample_weights, binning_expression):
        super().__init__()
        self.device = device
        self.data = data
        self.labels = torch.from_numpy(labels)
        self.meta = torch.from_numpy(meta) if meta is not None else None
        self.sample_weights = torch.tensor(sample_weights, dtype=torch.float)
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
                'x': full_seq.to(self.device),
                'y': self.labels[index].to(self.device),
                'meta': self.meta[index].to(self.device),
                'sample_weight': self.sample_weights[index].to(self.device)
            }
        else:
            return {
                'x': full_seq.to(self.device),
                'y': self.labels[index].to(self.device),
                'sample_weight': self.sample_weights[index].to(self.device)
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

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def main(rank: int, world_size: int, args: argparse.Namespace):
    ddp_setup(rank, world_size)
    device = torch.device("cuda:{}".format(rank))

    adata = sc.read_h5ad(args.data_path)
    if args.dataset == '':
        print(f'dataset is empty, use all data: {adata.X.shape}')
    else:
        data_list = args.dataset.split(",")
        data = adata[adata.obs['Dataset'].isin(data_list), :]
        adata = data
    if args.dataset_exclude != '':
        print(f'exclude datasets {args.dataset_exclude} for training')
        data_list = args.dataset_exclude.split(",")
        data = adata[~adata.obs['Dataset'].isin(data_list), :]
        adata = data
    if rank == 0:
        print(f'training data shape: {adata.X.shape}')

    if args.condition_embed:
        assert args.condition_column in adata.obs, f"can't find {args.condition_column} in data"
        condition_array, condition_train = np.unique(np.array(adata.obs[args.condition_column]), return_inverse=True)  # Convert strings categorical to integrate categorical, and celltypes_array[label] can be restored

        if rank == 0 and args.conditiontype_path != '':
            print(f'conditiontype_path:{args.conditiontype_path}')
            np.savetxt(args.conditiontype_path, condition_array, fmt="%s", delimiter=",")

        num_cond = len(condition_array)
    else:
        condition_train = None
        num_cond = 0
    
    if 'Celltype' in adata.obs:
        celltypes_array, label = np.unique(np.array(adata.obs['Celltype']), return_inverse=True)  # Convert strings categorical to integrate categorical, and celltypes_array[label] can be restored
    elif 'celltype' in adata.obs:
        celltypes_array, label = np.unique(np.array(adata.obs['celltype']), return_inverse=True)  # Convert strings categorical to integrate categorical, and celltypes_array[label] can be restored
    else:
        assert 0, "can't find Celltype or celltype in data"

    if rank == 0 and args.celltype_path != '':
        print(f'celltype_path:{args.celltype_path}')
        np.savetxt(args.celltype_path, celltypes_array, fmt="%s", delimiter=",")

    if args.apply_class_weight:
        classes = np.unique(label)
        # Compute class weights
        class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=label)
        print(class_weights)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
    else:
        class_weights_tensor = None
    if args.apply_sample_weight:
        sample_weights = compute_sample_weight(class_weight='balanced', y=label)
    else:
        sample_weights = np.ones(len(label))

    data = adata.X
    acc = []
    f1 = []
    train_dataset = SCDataset(device, data, label, condition_train, sample_weights, not args.mlp_embedding) 
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Must stay False when using a sampler! let sampler handle shuffling
        sampler=train_sampler
    )

    val_loader = None
    if args.data_path_val != '':
        adata_val = sc.read_h5ad(args.data_path_val)
        if args.dataset_val == '':
            print(f'dataset_val is empty, use all data in {args.data_path_val} for validation')
        else:
            data_list = args.dataset_val.split(",")
            data = adata_val[adata_val.obs['Dataset'].isin(data_list), :]
            adata_val = data
        if args.dataset_val_exclude != '':
            print(f'exclude datasets: {args.dataset_val_exclude} for validation')
            data_list = args.dataset_val_exclude.split(",")
            data = adata_val[~adata_val.obs['Dataset'].isin(data_list), :]
            adata_val = data
        if rank == 0:
            print(f'validation dataset: {adata_val.X.shape}')

        if args.condition_embed:
            condition_to_index = {label: idx for idx, label in enumerate(condition_array)}
            condition_val = adata_val.obs[args.condition_column].map(condition_to_index).to_numpy()
        else:
            condition_val = None

        label_to_index = {label: idx for idx, label in enumerate(celltypes_array)}
        label_val = adata_val.obs['Celltype'].map(label_to_index).to_numpy()

        data_val = adata_val.X
        if args.apply_sample_weight:
            sample_weights = compute_sample_weight(class_weight='balanced', y=label_val)
        else:
            sample_weights = np.ones(len(label_val))
        val_dataset = SCDataset(device, data_val, label_val, condition_val, sample_weights, not args.mlp_embedding) 
        val_sampler = DistributedSampler(val_dataset)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler)

    # for MLP embedding, don't add [cls] token
    seq_len = args.gene_num if args.mlp_embedding else args.gene_num + 1
    
    model = PlantLM(
        num_tokens = CLASS,
        dim = args.embed_dim,
        depth = args.num_layer,
        max_seq_len = seq_len,
        heads = args.heads,
        local_attn_heads = 0,
        mlp_embedding=args.mlp_embedding,
        ff_dropout=args.ff_dropout,
        gene_emb = args.gene_embed,
        condition_emb = args.condition_embed,
        num_conditions = num_cond
    )

    model.to_out = MLPClassifier(seq_len, args.embed_dim, args.h_dim, celltypes_array.shape[0], dropout=args.dropout)
    model = model.to(device)
    model = DDP(model, device_ids=[rank])
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    if args.scheduler_type == 'linear':
        scheduler = LambdaLR(optimizer, linear_warmup_decay(args.warmup_steps, args.epoch * len(train_loader) // args.grad_acc))
    elif args.scheduler_type == 'cosine':
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=args.first_cycle_steps,
            cycle_mult=2,
            max_lr=args.learning_rate,
            min_lr=1e-6,
            warmup_steps=args.warmup_steps,
            gamma=0.9
        )
    else:
        assert False, f"Unknown scheduler type: {args.scheduler_type}"
    start_epoch = 0
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, weight=class_weights_tensor, reduction = 'none' if args.apply_sample_weight else 'mean').to(device)

    if rank == 0:
        writer = SummaryWriter(log_dir=f"runs/{args.model_name}")  # You can change the log_dir name

    trigger_times = 0
    max_acc = 0.0
    max_f1 = 0.0
    softmax = nn.Softmax(dim=-1)
    for epoch in range(start_epoch + 1, args.epoch+1):
        train_loader.sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        cum_acc = 0.0

        if rank == 0:
            loop = tqdm(train_loader, desc=f"Epoch {epoch}")
        else:
            loop = train_loader

        for index, batch in enumerate(loop):
            index = index + 1
            global_step = epoch * len(train_loader) + index
            data = batch['x']
            labels = batch['y']
            weights = batch['sample_weight']
            meta = batch['meta'] if 'meta' in batch else None

            # Should sync only on last step of grad accumulation
            should_sync = index % args.grad_acc == 0 or index == len(train_loader)
            context = model.no_sync() if not should_sync else nullcontext()

            with context:
                logits = model(data) if meta is None else model.module.forward_with_meta(data, meta)
                loss = loss_fn(logits, labels)
                if args.apply_sample_weight:
                    weighted_loss = loss * weights
                    loss = weighted_loss.mean()
                loss.backward()
                running_loss += loss.item()
                if rank == 0:
                    loop.set_postfix(loss=loss.item())

            if should_sync:
                torch.nn.utils.clip_grad_norm_(model.parameters(), int(1e6))
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            running_loss += loss.item()
            final = softmax(logits).argmax(dim=-1)
            pred_num = labels.size(0)
            correct_num = torch.eq(final, labels).sum(dim=-1)
            acc = torch.true_divide(correct_num, pred_num).mean().item()
            cum_acc += acc

            if rank == 0 and index % args.log_every == 0:
                writer.add_scalar("Loss/Train_step", loss.item(), global_step)
                writer.add_scalar("Accuracy/Train_step", acc, global_step)
                writer.add_scalar("LR_step", optimizer.param_groups[0]['lr'], global_step)

        epoch_loss = running_loss / index
        epoch_acc = 100 * cum_acc / index
        epoch_loss = get_reduced(epoch_loss, rank, 0, world_size)
        epoch_acc = get_reduced(epoch_acc, rank, 0, world_size)
        if rank == 0:
            print(f'    ==  Epoch: {epoch} | Training Loss: {epoch_loss:.6f} | Accuracy: {epoch_acc:6.4f}%  ==')
            writer.add_scalar("Loss/Train", epoch_loss, epoch)
            writer.add_scalar("Accuracy/Train", epoch_acc, epoch)
            writer.add_scalar("LR", optimizer.param_groups[0]['lr'], epoch)

        if val_loader is not None and epoch % args.valid_every == 0:
            model.eval()
            running_loss = 0.0
            predictions = []
            truths = []
            if rank == 0:
                loop = tqdm(val_loader, desc=f"Validation Epoch {epoch}")
            else:
                loop = val_loader
            with torch.no_grad():
                for index, batch in enumerate(loop):
                    data_v = batch['x']
                    labels_v = batch['y']
                    weights_v = batch['sample_weight']
                    meta_v = batch['meta'] if 'meta' in batch else None
                    logits = model(data_v) if meta_v is None else model.module.forward_with_meta(data_v, meta_v)
                    loss = loss_fn(logits, labels_v)
                    if args.apply_sample_weight:
                        weighted_loss = loss * weights_v
                        loss = weighted_loss.mean()
                    running_loss += loss.item()
                    softmax = nn.Softmax(dim=-1)
                    probs = softmax(logits)
                    preds = probs.argmax(dim=-1)
                    preds[np.amax(np.array(probs.cpu()), axis=-1) < 0.0] = -1
                    predictions.append(preds)
                    truths.append(labels_v)
                del data_v, labels_v, weights_v, logits, probs, preds
                
                # Gather predictions and truths across ranks
                predictions = distributed_concat(torch.cat(predictions, dim=0), len(val_sampler.dataset), world_size)
                truths = distributed_concat(torch.cat(truths, dim=0), len(val_sampler.dataset), world_size)

                # Reduce validation loss across ranks
                val_loss = running_loss / index
                val_loss = get_reduced(val_loss, rank, 0, world_size)

                # Only rank 0 calculates metrics and makes stopping decision
                should_stop_tensor = torch.tensor([0], device=device)
                if rank == 0:
                    no_drop = predictions != -1
                    predictions = predictions[no_drop].cpu().numpy()
                    truths = truths[no_drop].cpu().numpy()
                    cur_acc = accuracy_score(truths, predictions)
                    f1 = f1_score(truths, predictions, average='macro')

                    print(f'    ==  Epoch: {epoch} | Validation Loss: {val_loss:.6f} | Acc: {cur_acc:.6f} | F1 Score: {f1:.6f}  ==')
                    writer.add_scalar("Loss/Validation", val_loss, epoch)
                    writer.add_scalar("Accuracy/Validation", cur_acc, epoch)
                    writer.add_scalar("F1/Validation", f1, epoch)

                    if f1 > max_f1:
                        max_f1 = f1
                        save_best_ckpt(epoch, model, optimizer, scheduler, val_loss, cur_acc, f1, celltypes_array.tolist(), args.model_name, args.ckpt_dir, tag='f1')
                    if cur_acc > max_acc:
                        max_acc = cur_acc
                        trigger_times = 0
                        save_best_ckpt(epoch, model, optimizer, scheduler, val_loss, cur_acc, f1, celltypes_array.tolist(), args.model_name, args.ckpt_dir)
                    else:
                        trigger_times += 1
                        if trigger_times > args.patience:
                            should_stop_tensor[0] = 1
                            print(f"Triggering early stopping at epoch {epoch}")
                # Broadcast stopping decision to all ranks
                torch.distributed.broadcast(should_stop_tensor, src=0)
                if should_stop_tensor.item() == 1:
                    break

        if epoch % args.save_every == 0 and rank == 0:
            save_ckpt(epoch, model, optimizer, scheduler, epoch_loss, epoch_acc, 0.0, celltypes_array.tolist(), args.model_name, args.ckpt_dir)

        if val_loader is None:  # if there is no validation set, use train set result to decide when to save model and abort training
            should_stop_tensor = torch.tensor([0], device=device)
            if rank == 0:
                if epoch_acc > max_acc:
                    max_acc = epoch_acc
                    trigger_times = 0
                    save_best_ckpt(epoch, model, optimizer, scheduler, epoch_loss, epoch_acc, 0.0, celltypes_array.tolist(), args.model_name, args.ckpt_dir)
                else:
                    trigger_times += 1
                    if trigger_times > args.patience:
                        should_stop_tensor[0] = 1
            # Broadcast stopping decision to all ranks
            torch.distributed.broadcast(should_stop_tensor, src=0)
            if should_stop_tensor.item() == 1:
                break

    if rank == 0:
        writer.close()
    # clean up
    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("distributed model training job")
    parser.add_argument("--bin_num", type=int, default=5, help='Number of bins.')
    parser.add_argument("--gene_num", type=int, default=20000, help='Number of genes.')
    parser.add_argument("--embed_dim", type=int, default=150, help='embedding dimension.')
    parser.add_argument("--heads", type=int, default=10, help='number of heads in Performer.')
    parser.add_argument("--num_layer", type=int, default=5, help='Number of transformer layers.')
    parser.add_argument("--h_dim", type=int, default=128, help='hidden dimension in MLP.')
    parser.add_argument("--epoch", type=int, default=100, help='Number of epochs.')
    parser.add_argument("--patience", type=int, default=4, help='number of epochs to wait before giving up.')
    parser.add_argument('--log_every', type=int, default=100, help='How often (# of steps) to log training loss and accuracy.')
    parser.add_argument('--save_every', type=int, default=100, help='How often (# of epochs) to save a snapshot in addition to best snapshot')
    parser.add_argument("--valid_every", type=int, default=1, help='Number of training epochs between twice validation.')
    parser.add_argument("--seed", type=int, default=42, help='Random seed.')
    parser.add_argument("--batch_size", type=int, default=3, help='Number of batch size.')
    parser.add_argument("--learning_rate", type=float, default=1e-4, help='Learning rate.')
    parser.add_argument("--scheduler_type", type=str, default='cosine', help='type of leearning rate scheduler. (cosine or linear)')
    parser.add_argument("--dropout", type=float, default=0.0, help='dropout rate in classifier.')
    parser.add_argument("--ff_dropout", type=float, default=0.0, help='dropout rate in transformer FF.')
    parser.add_argument("--label_smoothing", type=float, default=0.0, help='amount of label smoothing.')
    parser.add_argument("--warmup_steps", type=int, default=20, help='warm-up steps.')
    parser.add_argument("--first_cycle_steps", type=int, default=200, help='warm-up steps.')
    parser.add_argument("--grad_acc", type=int, default=60, help='Number of gradient accumulation.')
    parser.add_argument("--gene_embed", action="store_true", help='Using gene embedding or not.')
    parser.add_argument("--condition_embed", action="store_true", help='Using condition embedding or not.')
    parser.add_argument("--condition_column", type=str, default='Dataset', help='adata.obs column name for condition embedding.')
    parser.add_argument("--conditiontype_path", type=str, default='', help='condition type labels path')
    parser.add_argument("--data_path", type=str, default='all_tissues_hvg20k_train.h5ad', help='training h5ad data path.')
    parser.add_argument("--dataset", type=str, default='', help='dataset to use for training.')
    parser.add_argument("--dataset_exclude", type=str, default='', help='dataset to exclude from training.')
    parser.add_argument("--data_path_val", type=str, default='', help='Path of data for validation.')
    parser.add_argument("--dataset_val", type=str, default='all_tissues_hvg20k_val.h5ad', help='dataset to use for validation.')
    parser.add_argument("--dataset_val_exclude", type=str, default='', help='dataset to exclude from validation.')
    parser.add_argument("--model_path", type=str, default='', help='Path of pretrained model.')
    parser.add_argument("--ckpt_dir", type=str, default='.', help='Directory of checkpoint to save.')
    parser.add_argument("--model_name", type=str, default='unified', help='Finetuned model name.')
    parser.add_argument("--log_file", type=str, default='log.txt', help='log file path')
    parser.add_argument("--celltype_path", type=str, default='', help='celltype labels path')
    parser.add_argument("--update_allparams", action='store_true', help='update all model parameters.')
    parser.add_argument("--apply_class_weight", action='store_true', help='apply class weight in loss function.')
    parser.add_argument("--apply_sample_weight", action='store_true', help='apply sample weight in loss function.')
    parser.add_argument("--mlp_embedding", action='store_true', help='use mlp embedding.')

    args = parser.parse_args()

    print("Starting the training process")

    world_size = torch.cuda.device_count()
    print(f'num of GPUs: {world_size}')
    mp.spawn(main, args=(world_size, args), nprocs=world_size)

    print("Finished Training")
