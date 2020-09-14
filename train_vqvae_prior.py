import vqvae_prior

# ==============
# set-up
# ==============


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

img_dims = (3, 352, 352)[1:]  # image dimensions
input_dims = [img_dims, [(img_dims[0] // 4, img_dims[1] // 4)],
              [img_dims[0] // 8, img_dims[1] // 8]]  # input dims for the priors
n_bits = 9  # bits of the embedding dim of the codebook (2**n_bits = n_dims)

# ==============
# dataset
# ==============


codes = 'codes-vqvae.pt'
dataset = torch.load(codes)

loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, drop_last=True)


# prepare shape of the input encoding
def prepare(x):
    return x.unsqueeze_(-1).transpose(1, 3)


# ==============
# train
# ==============


epochs = 10


def load_model(hierarchy):
    if hierarchy == 'top':
        model = PixelSNAIL(input_dims=input_dims, n_channels=256, n_res_layers=5, n_out_stack_layers=5,
                           n_cond_classes=19000, n_bits=n_bits,
                           attn_n_layers=4, attn_nh=8, attn_dq=16, attn_dv=128, attn_drop_rate=0, drop_rate=0.1)

    elif hierarchy == 'bottom':
        model = PixelCNN(n_channels=256, n_out_conv_channels=1024, kernel_size=3, n_res_layers=20,
                         n_cond_stack_layers=10, n_cond_classes=19000, n_bits=n_bits,
                         drop_rate=0.1)

    return model


def train(model, hierarchy, epochs, lr, i):
    optimizer = optim.RMSprop(model.parameters(), lr=lr)
    model.train()

    pbar = tqdm(total=len(loader), position=0, leave=True, desc='epoch {}/{}'.format(i + 1, epochs))
    for bottom, top, caption in loader:

        bottom, top, caption = prepare(bottom).to(device), prepare(top).to(device), caption.to(device)

        if hierarchy == 'bottom':
            x = bottom
            out = model(preprocess(x, n_bits).float(), preprocess(top, n_bits=n_bits).float(), y=caption.float())
        elif hierarchy == 'top':
            x = top
            out = model(preprocess(x, n_bits=n_bits).float(), y=caption.float())

        loss = F.cross_entropy(out, x).mean(0)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), 1)
        optimizer.step()

        pbar.set_postfix(loss='{:.4f}'.format(loss.item() / np.log(2)))
        pbar.update()
        torch.cuda.empty_cache()

    if (i + 1) % 1 == 0:
        torch.save(
            {'model': model.state_dict()},
            f'checkpoint_pixelsnail_{hierarchy}_{str(i + 1).zfill(3)}.pt',
        )

#Train top prior
model=load_model('top')
model.to(device)

for i in range(epochs):
    train(model,'top',epochs,lr=1e-4,i=i)


#Train bottom prior
model=load_model('bottom')
model.to(device)

for i in range(epochs):
    train(model,'bottom',epochs,lr=1e-4,i=i)
