#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

from __future__ import print_function

import numpy as np

import umap

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid

from tqdm import tqdm

from math import sqrt

from VQVAE import VQVAE


# In[ ]:


# load model

model=VQVAE(in_channels=in_channels,num_hiddens=num_hiddens,num_residual_layers=num_residual_layers,num_residual_hiddens=num_residual_hiddens,num_embeddings=num_embeddings,
           embedding_dim=embedding_dim,commitment_cost=commitment_cost,decay=decay,)

ckpt=torch.load('vqvae2.pt')
model.load_state_dict(ckpt)
model.to(device)


# In[ ]:


@torch.no_grad()
def extract_codes(vqvae, dataloader, dataset_path):
#encode with vqvae and extract
    device = next(vqvae.parameters()).device
    bottoms, tops, captions = [], [], []
    for img, caption in tqdm(dataloader):
        img=img.to(device)
        _,_,_, id_top,id_bottom = vqvae.encode(img)
        
        bottoms.append(id_bottom.detach().cpu())
        tops.append(id_top.detach().cpu())
        captions.append(caption)
        del img 
        torch.cuda.empty_cache()
        
    return TensorDataset(torch.cat(bottoms), torch.cat(tops), torch.cat(captions))


def create_code_extract(vqvae, data):
    dataset_path = 'codes-vqvae.pt'
    if not os.path.exists(dataset_path):
        dataset = extract_codes(vqvae, data, dataset_path)
        torch.save(dataset, dataset_path)
    else:
        dataset = torch.load(dataset_path)
    return dataset


# In[ ]:


create_code_extract(model, dataloader)

