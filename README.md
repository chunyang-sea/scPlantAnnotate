# scPlantAnnotate
cell type annotation tool on scRNA-seq data

scPlantAnnotate is a deep learning framework for plant cell type annotation using scRNA-seq data across multiple species.

## Installation

```bash
git clone https://github.com/chunyang-sea/scPlantAnnotate.git
cd scPlantAnnotate

# Install PyTorch with CUDA 11.8 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies for data processing, ML
pip install pandas scanpy anndata scipy scikit-learn tensorboard einops local_attention
```

## Usage

```bash

# model training

# Read train.py for the full list of auguments and efault values

# very minimal usage: specify training data, validation data and output folder path
python train.py --data_path train.h5ad --data_path_val validation.h5ad --ckpt_dir checkpoints_dir

# use sample weighting
python train.py --data_path train.h5ad --data_path_val validation.h5ad --ckpt_dir checkpoints_dir --apply_sample_weight

# use label smoothing
python train.py --data_path train.h5ad --data_path_val validation.h5ad --ckpt_dir checkpoints_dir --label_smoothing 0.1

# use gene embedding (random initialized)
python train.py --data_path train.h5ad --data_path_val validation.h5ad --ckpt_dir checkpoints_dir --gene_embed

# use condition embedding, default condition column to use is adata.obs["Dataset"], change if needed
python train.py --data_path train.h5ad --data_path_val validation.h5ad --ckpt_dir checkpoints_dir --condition_embed --conditiontype_path output_conditions.list

# model evaluation

python eval.py --model_path checkpoint.pth --data_path test.h5ad

# test model trained with gene embedding
python eval.py --model_path checkpoint.pth --data_path test.h5ad --gene_embed

# test model trained with condition embedding
python eval.py --model_path checkpoint.pth --data_path test.h5ad --condition_embed --conditiontype_path output_conditions.list

```
## Pretrained model weights
```
pretrained scPlantAnnotate models can be downloaded from https://mailmissouri-my.sharepoint.com/:f:/g/personal/clcdp_umsystem_edu/Eu4e_4gD515Nv0YC4lxAyPkB9oroYJoJErM1P3vTwgd4Sg?e=AJbHh0

```

## Dataset information
```
information about dataset, tissue, celltypes that have been used to train scPlantAnnotate models are located in data folder
```
