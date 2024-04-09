# Our model was revised from D3DP, https://github.com/paTRICK-swk/D3DP, but finally we used the MixSTE only, removing everything about diffusion model
# And D3DP was revised from MixSTE, https://github.com/JinluZhang1126/MixSTE

## This following comment is from original PoseFormer model
## Our PoseFormer model was revised from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

import torch
import torch.nn as nn

from einops import rearrange
from functools import partial
from timm.models.layers import DropPath

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., changedim=False, currentdim=0, depth=0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., comb=False, vis=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim) 

        self.proj_drop = nn.Dropout(proj_drop)
        self.comb = comb
        self.vis = vis

    def forward(self, x, vis=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # Now x shape (3, B, heads, N, C//heads)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        if self.comb==True:
            attn = (q.transpose(-2, -1) @ k) * self.scale
        elif self.comb==False:
            attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        if self.comb==True:
            x = (attn @ v.transpose(-2, -1)).transpose(-2, -1)
            x = rearrange(x, 'B H N C -> B N (H C)')
        elif self.comb==False:
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., attention=Attention, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, comb=False, changedim=False, currentdim=0, depth=0, vis=False):
        super().__init__()

        self.changedim = changedim
        self.currentdim = currentdim
        self.depth = depth
        if self.changedim:
            assert self.depth>0

        self.norm1 = norm_layer(dim)
        self.attn = attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, comb=comb, vis=vis)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        if self.changedim and self.currentdim < self.depth//2:
            self.reduction = nn.Conv1d(dim, dim//2, kernel_size=1)
            # self.reduction = nn.Linear(dim, dim//2)
        elif self.changedim and depth > self.currentdim > self.depth//2:
            self.improve = nn.Conv1d(dim, dim*2, kernel_size=1)
            # self.improve = nn.Linear(dim, dim*2)
        self.vis = vis

    def forward(self, x, vis=False):
        x = x + self.drop_path(self.attn(self.norm1(x), vis=vis))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        if self.changedim and self.currentdim < self.depth//2:
            x = rearrange(x, 'b t c -> b c t')
            x = self.reduction(x)
            x = rearrange(x, 'b c t -> b t c')
        elif self.changedim and self.depth > self.currentdim > self.depth//2:
            x = rearrange(x, 'b t c -> b c t')
            x = self.improve(x)
            x = rearrange(x, 'b c t -> b t c')
        return x


class MultiModalGuesser(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,  norm_layer=None, is_train=True):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio   #### temporal embed_dim is num_joints * spatial embedding dim ratio
        out_dim = 3     #### output dimension is num_joints * 3
        self.is_train = is_train

        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)

        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))


        self.pos_drop = nn.Dropout(p=drop_rate)


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.block_depth = depth

        self.STEblocks = nn.ModuleList([
            # Block: Attention Block
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.TTEblocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, comb=False, changedim=False, currentdim=i+1, depth=depth)
            for i in range(depth)])

        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)

        self.mean_decoder = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim*2),
            nn.GELU(),
            nn.LayerNorm(embed_dim*2),
            nn.Linear(embed_dim*2, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
        )

        self.var_decoder = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim*2),
            nn.GELU(),
            nn.LayerNorm(embed_dim*2),
            nn.Linear(embed_dim*2, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1),
        )

        self.sample_decoder = nn.Sequential(
            nn.LayerNorm(embed_dim + 4),
            nn.Linear(embed_dim + 4, embed_dim*2),
            nn.GELU(),
            nn.LayerNorm(embed_dim*2),
            nn.Linear(embed_dim*2, embed_dim*4),
            nn.GELU(),
            nn.LayerNorm(embed_dim*4),
            nn.Linear(embed_dim*4, out_dim*10)
        )

    def STE_forward(self, x):

        b, f, n, c = x.shape
        x = rearrange(x, 'b f n c  -> (b f) n c', )
        x = self.Spatial_patch_to_embedding(x)
        x += self.Spatial_pos_embed

        x = self.pos_drop(x)

        blk = self.STEblocks[0]
        x = blk(x)

        x = self.Spatial_norm(x)
        x = rearrange(x, '(b f) n cw -> (b n) f cw', f=f)
        return x

    def TTE_foward(self, x):
        assert len(x.shape) == 3, "shape is equal to 3"
        b, f, _  = x.shape
        x += self.Temporal_pos_embed
        x = self.pos_drop(x)
        blk = self.TTEblocks[0]
        x = blk(x)

        x = self.Temporal_norm(x)
        return x

    def ST_foward(self, x):
        assert len(x.shape)==4, "shape is equal to 4"
        b, f, n, cw = x.shape
        for i in range(1, self.block_depth):
            x = rearrange(x, 'b f n cw -> (b f) n cw')
            steblock = self.STEblocks[i]
            tteblock = self.TTEblocks[i]
            
            x = steblock(x)
            x = self.Spatial_norm(x)
            x = rearrange(x, '(b f) n cw -> (b n) f cw', f=f)

            x = tteblock(x)
            x = self.Temporal_norm(x)
            x = rearrange(x, '(b n) f cw -> b f n cw', n=n)
        
        return x

    def forward(self, x_bar):
        b, f, n, c = x_bar.shape

        x = self.STE_forward(x_bar)
        x = self.TTE_foward(x)
        x = rearrange(x, '(b n) f cw -> b f n cw', n=n)
        x = self.ST_foward(x)

        mean = self.mean_decoder(x)
        var = self.var_decoder(x)
        sample = self.sample_decoder(torch.cat((x, mean, var), dim=-1))

        mean = mean.view(b, f, n, c)
        var = var.view(b, f, n, 1)
        sample = sample.view(b, 10, f, n, c)

        return mean, var, sample

    def reparameterize(self, mean, var, sample):
        extended_var = var[:, None, :, :, :] * torch.ones((mean.shape[0], 10, 1, 1, 1)).to(var.device)
        std = torch.exp(extended_var/2)
        std_sample = sample.std(dim=1).mean(dim=(1, 2, 3))
        sample_prediction = sample * std / std_sample[:, None, None, None, None]
        x_hat = sample_prediction + mean[:, None]

        return x_hat


if __name__ == '__main__':

    batch_size = 8
    num_frame = 25
    num_joints = 16
    in_chans = 3

    model = MultiModalGuesser(num_frame=num_frame, num_joints=num_joints, in_chans=in_chans, embed_dim_ratio=32, depth=4,)

    x = torch.randn(batch_size, num_frame, num_joints, in_chans)
    mean, var, sample = model(x)
    print(f"mean shape: {mean.shape}, var shape: {var.shape}, sample shape: {sample.shape}")

    extended_var = var[:, None, :, :, :] * torch.ones((x.shape[0], 10, 1, 1, 1))
    std = torch.exp(extended_var/2)
    std_sample = sample.std(dim=1).mean(dim=(1, 2, 3))
    sample_prediction = sample * std / std_sample[:, None, None, None, None]
    x_hat = sample_prediction + mean[:, None]

    print(x_hat.shape)

    gt = torch.randn(batch_size, num_frame, num_joints, in_chans)
    difference = gt[:, None] - x_hat
    loss = torch.mean(difference**2)
