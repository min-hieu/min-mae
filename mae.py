import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from attn import ResidualAttentionBlock, AttentionPooling
from einops import rearrange
from tqdm import tqdm, trange


class MAE(nn.Module):
    def __init__(
        self,
        patch_dim = 14,
        enc_embed_dim = 1024,
        enc_layers = 4,
        enc_heads = 8,
        dec_embed_dim = 512,
        dec_layers = 4,
        dec_heads = 8,
        mask_ratio = 0.75,
    ):
        super().__init__()
        self.patch_dim = patch_dim
        self.mask_ratio = mask_ratio
        self.enc_embed_dim = enc_embed_dim
        self.dec_embed_dim = dec_embed_dim
        self.enc_head_dim = enc_embed_dim // enc_heads
        self.dec_head_dim = dec_embed_dim // dec_heads

        self.enc_abp = nn.Parameter(torch.randn(256, enc_embed_dim))
        self.dec_abp = nn.Parameter(torch.randn(256, dec_embed_dim))

        self.patch_embed = nn.Conv2d(3, enc_embed_dim, patch_dim, patch_dim)
        self.enc_blk = nn.ModuleList([
            ResidualAttentionBlock(enc_embed_dim, enc_heads) for _ in range(enc_layers)
        ])
        self.mask_tok = nn.Parameter(torch.randn(dec_embed_dim))
        self.enc2dec = nn.Linear(enc_embed_dim, dec_embed_dim)
        self.dec_blk = nn.ModuleList([
            ResidualAttentionBlock(dec_embed_dim, dec_heads) for _ in range(dec_layers)
        ])
        self.head = nn.Linear(dec_embed_dim, patch_dim*patch_dim*3)

    def forward(self, x, return_mask=False):
        B,C,H,W = x.shape
        pH, pW = H//self.patch_dim, W//self.patch_dim
        assert H % self.patch_dim == 0 and W % self.patch_dim == 0

        x = self.patch_embed(x)
        x = x.permute(0,2,3,1)

        msk = torch.rand((pH, pW), device=x.device) < self.mask_ratio
        while not msk.any():
            msk = torch.rand((B, pH, pW)) > self.mask_ratio
        msk_nz = msk.count_nonzero()
        msk = msk.view(1, pH, pW).repeat(B, 1, 1)

        x += self.enc_abp.view(1, 256, -1).expand(B, 256, -1).view(B, 16, 16, -1)
        x_enc = x[msk].reshape(B, msk_nz, -1)

        for blk in self.enc_blk:
            x_enc = blk(x_enc)

        x_enc = self.enc2dec(x_enc).view(B*msk_nz, -1)
        x_msk = self.mask_tok.view(1,-1).expand(B*(pH*pW - msk_nz), -1)

        x_dec = torch.zeros(B, pH*pW, self.dec_embed_dim, device=x.device)
        msk = msk.reshape(-1, pH*pW)
        x_dec[msk] = x_enc
        x_dec[~msk] = x_msk

        x_dec += self.dec_abp.view(1, 256, -1).expand(B, 256, -1)

        for blk in self.dec_blk:
            x_dec = blk(x_dec)

        x_dec.view(B, pH, pW, -1)
        x_out = self.head(x_dec).view(B, pH, pW, self.patch_dim, self.patch_dim, 3)
        x_out = x_out.permute(0,5,1,3,2,4).reshape(B, 3, H, W)
        if return_mask:
            return x_out, ~msk[0].reshape(pH, pW)
        return x_out


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
dset = DataLoader(
    datasets.ImageFolder(
        "./data/imagenet_subtrain",
        transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,std=std)
    ])),
    batch_size=8,
    shuffle=True,
    num_workers=8,
    pin_memory=True
)
dset_iter = iter(dset)

device = torch.device(0)

model = MAE().to(device)
optim = torch.optim.AdamW(model.parameters(), lr=1e-5)

n_iter = 15000
pbar = trange(n_iter)
losses = []
for i in pbar:
    try:
        img, _ = next(dset_iter)
    except:
        dset_iter = iter(dset)
        img, _ = next(dset_iter)

    img = img.to(device)

    img_recon = model(img)
    loss = F.mse_loss(img_recon, img)
    loss.backward()
    optim.step()
    optim.zero_grad()

    if (i + 1) % 10 == 0 and i > 10:
        pbar.set_description(f"loss: {np.array(losses)[-10:].mean():.4f}")

    if (i + 1) % 100 == 0:
        losses.append(loss.item())

import matplotlib.pyplot as plt
# plt.plot(losses)
# plt.show()
# import pdb; pdb.set_trace()

def for_vis(img):
    img = img.permute(1,2,0).cpu().numpy() * np.array(std) + np.array(mean)
    return img.clip(0,1)

torch.save(losses, f"losses_{n_iter}.pth")
torch.save(model.state_dict(), f"mae_{n_iter}.pth")

model = MAE().to(device)
model.load_state_dict(torch.load(f"mae_{n_iter}.pth"))

try:
    img, _ = next(dset_iter)
except:
    dset_iter = iter(dset)
    img, _ = next(dset_iter)
img = img.to(device)
with torch.no_grad():
    img_recon, msk = model(img, return_mask=True)
    img_msk = img[0].clone()
    pH, pW = msk.shape
    msk = msk.reshape(pH, 1, pW, 1).expand(pH, 14, pW, 14).reshape(pH*14, pW*14)
    img_msk.permute(1,2,0)[msk] = 0
    # img_msk = msk.view(1, pH*14, pW*14).repeat(3,1,1)

fig, ax = plt.subplots(1,3)
ax[0].imshow(for_vis(img[0]))
ax[1].imshow(for_vis(img_msk))
ax[2].imshow(for_vis(img_recon[0]))
plt.show()
