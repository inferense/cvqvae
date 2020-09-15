# Text-to-image synthesis using CVQVAE
## Pytorch implementation of conditional-VQVAE2 for generating high-fidelity multi-object images based on text captions.

original paper: [Generating Diverse High-Fidelity Images with VQ-VAE-2](https://arxiv.org/abs/1906.00446)

This implementation is optimized for the MS-COCO dataset (Captions 2014). Currently supports hierarchical VQVAE and PixelSNAIL.

The code was imported from ipynb notebook and is in post processing. 

Credits: vqvae_prior.py code adapted from kamenbliznashki

### Preprequisites 
> - Downloaded MS-COCO captions dataset
> - Pytorch >= 1.6
> - GPU environment - the PixelSNAIL (vqvae_prior.py) is heavy to train especially on high-resolution images


### Usage
  1. Train vqvae.py
  2. extract codes
  3. Train vqvae_prior.py
  4. Sample 
  

