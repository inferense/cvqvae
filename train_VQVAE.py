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


#================
# Hyperparameters
#================


# In[ ]:


epochs=200

num_hiddens=128
num_residual_hiddens=32
num_residual_layers=2
embedding_dim=64
num_embeddings=512

in_channels=3

commitment_cost=0.25
decay=0.99
lr=3e-4


# In[ ]:


model=VQVAE(in_channels=in_channels,num_hiddens=num_hiddens,num_residual_layers=num_residual_layers,num_residual_hiddens=num_residual_hiddens,num_embeddings=num_embeddings,
           embedding_dim=embedding_dim,commitment_cost=commitment_cost,decay=decay,)

model.to(device)

optimizer=optim.Adam(model.parameters(), lr=lr,amsgrad=False)


# In[ ]:


def train(epoch, model, loader, optimizer, device):
    loader=tqdm(loader)
    criterion=nn.MSELoss()
    
    sample_size=25
    MSE_sum=0
    MSE_n=0
    latent_loss_weight=0.25

    for i,(img,label) in enumerate(loader):
        optimizer.zero_grad()
        img=img.to(device)
        
        out, latent_loss=model(img)
        recon_loss=criterion(out,img)
        latent_loss=latent_loss.mean()
        loss=recon_loss+latent_loss_weight*latent_loss
        loss.backward()

        optimizer.step()
        
        MSE_sum+=(recon_loss.item()*img.shape[0])
        MSE_n+=img.shape[0]
        
        loader.set_description(
            (
            f'epoch: {epoch+1};'
            f'MSE: {recon_loss.item():.5f}; latent_loss: {latent_loss:.5f};'
            f'avg_MSE:{MSE_sum/MSE_n:.5f}'
            )
        )


# In[ ]:


for i in range(epochs):
    train(i, model,dataloader,optimizer,device)


# In[ ]:


filepath='vqvae2.pt'
torch.save(model.state_dict(),filepath)


# In[ ]:


model=VQVAE(in_channels=in_channels,num_hiddens=num_hiddens,num_residual_layers=num_residual_layers,num_residual_hiddens=num_residual_hiddens,num_embeddings=num_embeddings,
           embedding_dim=embedding_dim,commitment_cost=commitment_cost,decay=decay,)

ckpt=torch.load('vqvae2.pt')
model.load_state_dict(ckpt)
model.to(device)


# In[ ]:


#==============
# show reconstructions
#==============


# In[ ]:


(valid_originals, _) = next(iter(testloader))
valid_originals = valid_originals.to(device)

# 1. encode (pre_conv,quantize) 2. decode (quantized)
valid_reconstruction,_ = model(valid_originals)


# In[ ]:


(train_originals, _) = next(iter(trainloader))
train_originals = train_originals.to(device)

train_reconstruction,_=model(train_originals)


# In[ ]:


def show(img,title):
    npimg = img.numpy()
    plt.figure(figsize=(25,25))
    fig = plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.title(title)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)


# In[ ]:


show(make_grid(valid_originals.cpu()+0.5),title='originals')


# In[ ]:


show(make_grid(valid_reconstruction.cpu().data)+0.5,title='reconstructed')

