import vqvae
import vqvae_prior
from torch.autograd import Variable

def sample_prior(model, h, y, n_samples, in_dims, n_bits):
    model.eval()

    H, W = in_dims
    out = torch.zeros(1, 1, H, W, device=next(model.parameters()).device)
    pbar = tqdm(total=H * W, desc='Generating {} images'.format(n_samples))
    y = y.unsqueeze(1)
    y = y.to(device)
    for hi in range(H):
        for wi in range(W):
            logits = model(out.float(), y.float()) if h is None else model(out.float(), h.float(), y.float())
            probs = F.softmax(logits, dim=1)
            sample = torch.multinomial(probs[:, :, :, hi, wi].squeeze(2), 1)
            out[:, :, hi, wi] = preprocess(sample,
                                           n_bits)  # multinomial samples long tensor in [0, 2**n_bits), convert back to model space [-1,1]
            pbar.update()
            del logits, probs, sample
    pbar.close()
    return deprocess(out, n_bits)  # out (B,1,H,W) field of latents in latent space [0, 2**n_bits)


@torch.no_grad()
def generate(vqvae, bottom_prior, top_prior, y=None):
        # sample top prior conditioned on class labels y
    top_sample = sample_prior(top_prior, None, y, n_samples=1, in_dims=input_dims[2], n_bits=n_bits)
        # sample bottom prior conditioned on top_sample codes and class labels y
    bottom_sample = sample_prior(bottom_prior, preprocess(top_samples, n_bits=9), y, n_samples=1, in_dims=input_dims[1], n_bits=n_bits)
        #decode
    decoded_sample = vqvae.decode_code(top_samples.squeeze(1),bottom_samples.squeeze(1))
    decoded_sample=decoded_sample.clamp(-1,1)

    return save_image(decoded_sample,'sample2.png',normalize=True,range=(-1,1))



def load_model(model):
    if model == 'vqvae':
        model = VQVAE(in_channels=3, num_hiddens=128, num_residual_layers=2, num_residual_hiddens=32,
                      num_embeddings=512,
                      embedding_dim=64, commitment_cost=0.25, decay=0.99, )
        ckpt = torch.load('vqvae2.pt')

    elif model == 'top_prior':
        model = PixelSNAIL(input_dims=input_dims, n_channels=256, n_res_layers=5, n_out_stack_layers=5,
                           n_cond_classes=19000, n_bits=n_bits,
                           attn_n_layers=4, attn_nh=8, attn_dq=16, attn_dv=128, attn_drop_rate=0, drop_rate=0.1)
        ckpt = torch.load('checkpoint_pixelsnail_top_005.pt')

    elif model == 'bottom_prior':
        model = PixelCNN(n_channels=256, n_out_conv_channels=1024, kernel_size=3, n_res_layers=20,
                         n_cond_stack_layers=10, n_cond_classes=19000, n_bits=n_bits,
                         drop_rate=0.1)
        ckpt = torch.load('bottom_prior.pt')

    model.load_state_dict(ckpt, strict=False)
    model = model.to(device)
    model.eval()

    return model


def input_y():
    y=input('input y:')
    y=list(y)
    y=V.transform(y)
    y=y.toarray()
    y=Variable(torch.from_numpy(y))
    return y


vqvae=load_model('vqvae')
top_prior=load_model('top_prior')
bottom_prior=load_model('bottom_prior')

samples = generate(vqvae, bottom_prior, top_prior, y=y)
