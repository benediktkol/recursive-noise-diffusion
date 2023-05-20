import math
import torch
import torch.nn.functional as F

from einops import rearrange
from torch import nn, einsum


def exists(x):
    return x is not None

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim_in, dim_out):
    return nn.ConvTranspose2d(dim_in, dim_out, 4, 2, 1)

def Downsample(dim_in, dim_out):
    return nn.Conv2d(dim_in, dim_out, 4, 2, 1)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class ResNetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""
    
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) #if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)

        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            h = rearrange(time_emb, "b c -> b c 1 1") + h

        h = self.block2(h)
        return h + self.res_conv(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), 
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class NetworkConfig:
    """Configuration for the network."""
    # Default configuration
    image_channels=3
    n_classes=19
    dim=32
    dim_mults=(1, 2, 4, 8)
    resnet_block_groups=8

    # diffusion parameters
    n_timesteps = 10
    n_scales = 3
    max_patch_size = 512
    scale_procedure = "loop" # "linear" or "loop"

    # ensemble parameters
    built_in_ensemble = False

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)


class Network(nn.Module):
    def __init__(
            self,
            network_config=NetworkConfig(),
            ): 
        super().__init__()
        self.config = network_config
        image_channels = self.config.image_channels
        n_classes = self.config.n_classes
        dim = self.config.dim
        dim_mults = self.config.dim_mults
        resnet_block_groups = self.config.resnet_block_groups

        # determine dimensions
        self.image_channels = image_channels
        self.n_classes = n_classes
        self.dims = [c * dim for c in dim_mults]

        # time embedding
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # image initial 
        self.image_initial = nn.ModuleList([
            ResNetBlock(image_channels, self.dims[0], time_emb_dim=time_dim, groups=resnet_block_groups),
            ResNetBlock(self.dims[0], self.dims[0], groups=resnet_block_groups),
            ResNetBlock(self.dims[0], self.dims[0], groups=resnet_block_groups)
        ])

        # segmentation initial 
        self.seg_initial = nn.ModuleList([
            ResNetBlock(n_classes, self.dims[0], time_emb_dim=time_dim, groups=resnet_block_groups),
            ResNetBlock(self.dims[0], self.dims[0], groups=resnet_block_groups),
            ResNetBlock(self.dims[0], self.dims[0], groups=resnet_block_groups)
        ])

        # layers
        self.down = nn.ModuleList([])
        self.up = nn.ModuleList([])

        # encoder
        for i in range(len(dim_mults)-1): # each dblock
            dim_in = self.dims[i]
            dim_out = self.dims[i+1]

            self.down.append(
                nn.ModuleList([
                    ResNetBlock(dim_in, dim_in, time_emb_dim=time_dim, groups=resnet_block_groups),
                    ResNetBlock(dim_in, dim_in, groups=resnet_block_groups),
                    Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                    Downsample(dim_in, dim_out),
                    
                ])
            )
                
        # decoder
        for i in range(len(dim_mults)-1): # each ublock
            dim_in = self.dims[-i-1]
            dim_out = self.dims[-i-2]
            if i == 0:
                dim_in_plus_concat = dim_in
            else:
                dim_in_plus_concat = dim_in * 2
            
            self.up.append(
                nn.ModuleList([
                    ResNetBlock(dim_in_plus_concat, dim_in, time_emb_dim=time_dim, groups=resnet_block_groups),
                    ResNetBlock(dim_in, dim_in, groups=resnet_block_groups),
                    Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                    Upsample(dim_in, dim_out),
                ])
            )

        # final
        self.final = nn.Sequential(ResNetBlock(self.dims[0]*2, self.dims[0], groups=resnet_block_groups), 
                                   ResNetBlock(self.dims[0], self.dims[0], groups=resnet_block_groups),
                                   nn.Conv2d(self.dims[0], n_classes, 1))



    def forward(self, seg, img, time):
        # time embedding
        t = self.time_mlp(time)

        # segmentation initial
        resnetblock1, resnetblock2, resnetblock3 = self.seg_initial
        seg_emb = resnetblock1(seg, t)
        seg_emb = resnetblock2(seg_emb)
        seg_emb = resnetblock3(seg_emb)

        # image initial
        resnetblock1, resnetblock2, resnetblock3 = self.image_initial
        img_emb = resnetblock1(img, t)
        img_emb = resnetblock2(img_emb)
        img_emb = resnetblock3(img_emb)

        # add embeddings together
        x = seg_emb + img_emb
        
        # skip connections
        h = []

        # downsample
        for resnetblock1, resnetblock2, attn, downsample in self.down:
            x = resnetblock1(x, t)
            x = resnetblock2(x)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        # upsample
        for  resnetblock1, resnetblock2, attn, upsample in self.up:
            x = resnetblock1(x, t)
            x = resnetblock2(x)
            x = attn(x)
            x = upsample(x)
            x = torch.cat((x, h.pop()), dim=1)

        return self.final(x)