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

import os

from collections import namedtuple
from torch.utils.data import DataLoader, TensorDataset

import torchvision.utils as vutils

from sklearn.feature_extraction.text import TfidfVectorizer
import json


# In[ ]:


#====================
# Dataset
#====================


# In[ ]:


from pycocotools.coco import COCO


# In[ ]:


annfile='coco/annotations/captions_train2014.json' # annotations directory


# In[ ]:


#coco=COCO(annfile)


# In[ ]:


with open(annfile) as json_file:
    data=json.load(json_file)


# In[ ]:


corpus=[]
dicts=data['annotations']
for k in dicts:
    corpus.append(k['caption'])


# In[ ]:


max_features=19000 #restrict size of the corpus 
V=TfidfVectorizer(use_idf=True,smooth_idf=True, max_features=max_features,stop_words='english')


# In[ ]:


V.fit(corpus)


# In[ ]:


V.idf_.shape


# In[ ]:


crop=500
img_size=352
workers=0
batch_size=1


# In[ ]:


def tmp_func(y):
    return V.transform(y[:1]).toarray()


# In[ ]:


dataset = datasets.CocoCaptions(root ='coco/images/train2014',
                        annFile = annfile,
                        transform=transforms.Compose([transforms.CenterCrop(crop),
                                                      transforms.Resize(img_size),  
                                                      transforms.ToTensor(),
                                                      transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                        ]),
                               target_transform=transforms.Lambda(tmp_func)
                               )

# train_size=int(0.7*(len(dataset)))
# test_size=len(dataset)-train_size
# train_dataset, test_dataset=torch.utils.data.random_split(dataset, [train_size,test_size])


# In[ ]:


dataloader=torch.utils.data.DataLoader(dataset,batch_size=batch_size, shuffle=True,pin_memory=True,num_workers=workers)
# trainloader=torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,pin_memory=True, num_workers=workers)
# testloader=torch.utils.data.DataLoader(test_dataset, batch_size=8,shuffle=True,pin_memory=True,num_workers=workers)


# In[ ]:


device=torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
device


# In[ ]:


# show test batch
idx=random.randint(0,3000)

image=dataset.__getitem__(idx)[0]
label=dataset.__getitem__(idx)[1]

plt.figure(figsize=(10,10))
plt.imshow(np.transpose(image.cpu().detach().numpy(), (1, 2, 0)))
plt.title(label)


# In[ ]:


images= next(iter(testloader))

plt.figure(figsize=(30,30))
plt.title('test batch')
plt.imshow(np.transpose(vutils.make_grid(images[0].to(device)[:3],padding=2,normalize=True).cpu(),(1,2,0)))
# print(labels[0][:3])


# In[ ]:


# data_variance = np.var(training_data.data / 255.0)


# In[ ]:


# variance=0.
# for image,label in dataset:
#     variance+=(image.var()/255.0)
# variance/=len(dataset)


# In[ ]:


#variance=(0.0011)


# In[ ]:


#================
# VQ
#================


# In[ ]:


class Quantize(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, decay=0.99, epsilon=1e-5):
        super().__init__()
        
        self.embedding_dim=embedding_dim
        self.num_embeddings=num_embeddings
        self.decay=decay
        self.epsilon=epsilon
        
        embedding=torch.randn(embedding_dim,num_embeddings)
        self.register_buffer('embedding',embedding)
        self.register_buffer('cluster_size',torch.zeros(num_embeddings))
        self.register_buffer('embeddings_avg', embedding.clone())
        
    def forward(self, inputs):
        flatten=inputs.reshape(-1, self.embedding_dim)
        
        #compute euclidian distance between latents and embeddings  
        distance=(
            flatten.pow(2).sum(1,keepdim=True) 
            -2 * flatten @ self.embedding
            + self.embedding.pow(2).sum(0,keepdim=True)
        )
        
        #get the min distance 
        _,embedding_indices=(-distance).max(1)
        
        #tranform to one-hot embeddings
        embeddings_onehot=F.one_hot(embedding_indices, self.num_embeddings).type(flatten.dtype)
        embedding_indices=embedding_indices.view(*inputs.shape[:-1])
        quantize=self.embedding_code(embedding_indices)
        
        if self.training:
            embeddings_onehot_sum=embeddings_onehot.sum(0)
            embeddings_sum=flatten.transpose(0,1)@embeddings_onehot
            
            self.cluster_size.data.mul_(self.decay).add_(
                embeddings_onehot_sum, alpha=1-self.decay
            )
            
            self.embeddings_avg.data.mul_(self.decay).add_(embeddings_sum, alpha=1-self.decay)
            n=self.cluster_size.sum()
            cluster_size=(
                (self.cluster_size+self.epsilon) / (n+self.num_embeddings*self.epsilon)*n
            )
            
            embeddings_normalized=self.embeddings_avg/cluster_size.unsqueeze(0)
            self.embedding.data.copy_(embeddings_normalized)
            
        diff=(quantize.detach()-inputs).pow(2).mean()
        quantize=inputs+(quantize-inputs).detach()
        
        ######
#         avg_probs=torch.mean(embeddings_onehot, dim=0)
#         perplexity=torch.exp(-torch.sum(avg_probs*torch.log(avg_probs+1e-10)))
        
        return quantize, diff, embedding_indices, 
    
    
    def embedding_code(self, embedding_id):
        return F.embedding(embedding_id,self.embedding.transpose(0,1))
        


# In[ ]:


class Resnet(nn.Module):
    def __init__(self,in_channels,num_hiddens,num_residual_hiddens):
        super().__init__()
        self._block=nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,out_channels=num_residual_hiddens,kernel_size=3,stride=1,padding=1,bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens, out_channels=num_hiddens,kernel_size=1,stride=1,bias=False)
        )
        
    def forward(self,x):
        return x+self._block(x)


class ResnetStack(nn.Module):
    def __init__(self,in_channels,num_hiddens,num_residual_layers,num_residual_hiddens):
        super(ResnetStack,self).__init__()
        self.num_residual_layers=num_residual_layers
        self.layers=nn.ModuleList([Resnet(in_channels,num_hiddens,num_residual_hiddens)
                                   for _ in range(self.num_residual_layers)])
        
    def forward(self,x):
        for i in range(self.num_residual_layers):
            x=self.layers[i](x)
        return F.relu(x)
        


# In[ ]:


class Encoder(nn.Module):
    def __init__(self, in_channels,num_hiddens,num_residual_layers, num_residual_hiddens, stride):
        super().__init__()
        
        if stride==4:
            stack=[
                nn.Conv2d(in_channels=in_channels, out_channels=num_hiddens//2, kernel_size=4, stride=2, padding=1),
                nn.ReLU(True),
                nn.Conv2d(in_channels=num_hiddens//2,out_channels=num_hiddens, kernel_size=4,stride=2,padding=1),
                nn.ReLU(True),
                nn.Conv2d(in_channels=num_hiddens,out_channels=num_hiddens,kernel_size=3,stride=1,padding=1),
                ResnetStack(in_channels=num_hiddens, num_hiddens=num_hiddens,num_residual_layers=num_residual_layers,
                           num_residual_hiddens=num_residual_hiddens)
            ]
        
        elif stride==2:
            stack=[
                nn.Conv2d(in_channels=in_channels,out_channels=num_hiddens//2,kernel_size=4,stride=2,padding=1),
                nn.ReLU(True),
                nn.Conv2d(in_channels=num_hiddens//2,out_channels=num_hiddens,kernel_size=3,stride=1,padding=1),
                ResnetStack(in_channels=num_hiddens, num_hiddens=num_hiddens,num_residual_layers=num_residual_layers,
                           num_residual_hiddens=num_residual_hiddens)
            ]
            
        self.stack=nn.Sequential(*stack)
        
    def forward(self,inputs):
        return self.stack(inputs)


# In[ ]:


class Decoder(nn.Module):
    def __init__(self, in_channels,out_channels,num_hiddens,num_residual_layers,num_residual_hiddens,stride):
        super().__init__()
        
        stack=[
            nn.Conv2d(in_channels=in_channels,out_channels=num_hiddens, kernel_size=3, stride=1,padding=1),
            ResnetStack(in_channels=num_hiddens,num_hiddens=num_hiddens, num_residual_layers=num_residual_layers,
                       num_residual_hiddens=num_residual_hiddens)
              ]
        
        if stride==4:
            stack.extend([
                nn.ConvTranspose2d(in_channels=num_hiddens,out_channels=num_hiddens//2,kernel_size=4,stride=2,padding=1),
                nn.ReLU(True),
                nn.ConvTranspose2d(in_channels=num_hiddens//2,out_channels=out_channels,kernel_size=4,stride=2,padding=1),
            ])
        
        elif stride==2:
            stack.append(
                nn.ConvTranspose2d(in_channels=num_hiddens,out_channels=out_channels,kernel_size=4,stride=2,padding=1)
            )
            
        self.stack=nn.Sequential(*stack)
        
    def forward(self,inputs):
        return self.stack(inputs)


# In[ ]:


class VQVAE(nn.Module):
    def __init__(self,in_channels, num_hiddens, num_residual_layers,num_residual_hiddens,num_embeddings,embedding_dim,commitment_cost,decay,):
        super().__init__()
        
        self.encode_bottom=Encoder(in_channels, num_hiddens, num_residual_layers,num_residual_hiddens,stride=4)
        self.encode_top=Encoder(num_hiddens,num_hiddens,num_residual_layers,num_residual_hiddens,stride=2)
        
        self.quantize_conv_top=nn.Conv2d(num_hiddens,embedding_dim,1)
        self.quantize_top=Quantize(embedding_dim, num_embeddings)
        self.decode_top=Decoder(embedding_dim,embedding_dim,num_hiddens,num_residual_layers,num_residual_hiddens,stride=2)
        
        self.quantize_conv_bottom=nn.Conv2d(embedding_dim+num_hiddens,embedding_dim,1)
        self.quantize_bottom=Quantize(embedding_dim,num_embeddings)
        
        self.unsample_top=nn.ConvTranspose2d(embedding_dim,embedding_dim, kernel_size=4,stride=2,padding=1)
        self.decode_=Decoder(embedding_dim+embedding_dim,in_channels,num_hiddens,num_residual_layers,num_residual_hiddens,stride=4)
        
        
    def forward(self,inputs):
        quantized_top,quantized_bottom,diff,_,_=self.encode(inputs)
        decode_=self.decode(quantized_top,quantized_bottom)
        
        return decode_,diff
    
    
    def encode(self,inputs):
        encode_bottom=self.encode_bottom(inputs)
        encode_top=self.encode_top(encode_bottom)
        
        quantized_top=self.quantize_conv_top(encode_top).permute(0,2,3,1)
        quantized_top,diff_top,id_top=self.quantize_top(quantized_top)
        quantized_top=quantized_top.permute(0,3,1,2)
        diff_top=diff_top.unsqueeze(0)
        
        decode_top=self.decode_top(quantized_top)
        encode_bottom=torch.cat([decode_top,encode_bottom],1)
            
        quantized_bottom=self.quantize_conv_bottom(encode_bottom).permute(0,2,3,1)
        quantized_bottom,diff_bottom,id_bottom=self.quantize_bottom(quantized_bottom)
        quantized_bottom=quantized_bottom.permute(0,3,2,1)
        diff_bottom=diff_bottom.unsqueeze(0)
        
        return quantized_top,quantized_bottom,diff_top+diff_bottom,id_top,id_bottom
    
    
    def decode(self,quantized_top,quantized_bottom):
        unsample_top=self.unsample_top(quantized_top)
        quantized=torch.cat([unsample_top,quantized_bottom],1)
        decode_=self.decode_(quantized)
        
        return decode_
        
        
    def decode_code(self,code_top,code_bottom):
        quantized_top=self.quantize_top.embedding_code(code_top)
        quantized_top=quantized_top.permute(0,3,1,2)
        quantized_bottom=self.quantize_bottom.embedding_code(code_bottom)
        quantized_bottom=quantized_bottom.permute(0,3,1,2)
        
        decode_=self.decode(quantized_top,quantized_bottom)
        
        return decode_

