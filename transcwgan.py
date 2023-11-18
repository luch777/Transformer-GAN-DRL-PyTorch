import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_feat, hid_feat=None, out_feat=None, dropout=0.):
        super().__init__()
        if not hid_feat:
            hid_feat = in_feat
        if not out_feat:
            out_feat = in_feat
        self.fc1 = nn.Linear(in_feat, hid_feat)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hid_feat, out_feat)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return self.dropout(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, attention_dropout=0., proj_dropout=0.):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.scale = 1. / dim ** 0.5

        self.q = nn.Linear(dim, dim * self.heads)
        self.k = nn.Linear(dim, dim * self.heads)
        self.v = nn.Linear(dim, dim * self.heads)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(proj_dropout)
        )

    def forward(self, x):
        b, n, c = x.shape
        q = self.q(x).reshape(b, n, self.dim, self.heads)
        k = self.k(x).reshape(b, n, self.dim, self.heads)
        v = self.v(x).reshape(b, n, self.dim, self.heads)

        # dot = (q @ k.transpose(-2, -1)) * self.scale  # @为矩阵乘法
        dot = q * k * self.scale
        attn = dot.softmax(dim=-1)
        attn = self.attention_dropout(attn)

        x = attn * v
        x = torch.mean(x, dim=3).reshape(b, n, c)
        x = self.out(x)
        return x


class ImgPatches(nn.Module):
    def __init__(self, patch_size=4):
        super().__init__()
        self.patch_embed = nn.Conv2d(1, 1, patch_size, patch_size)

    def forward(self, img):
        patches = self.patch_embed(img)  # 16维
        return patches.flatten(2)


class TransformerEncoder(nn.Module):
    def __init__(self, dim=16, heads=4, mlp_ratio=4, drop_rate=0.):
        super().__init__()
        self.dim = dim
        self.ln1 = nn.LayerNorm(self.dim)
        self.attn = Attention(self.dim, heads, drop_rate, drop_rate)
        self.ln2 = nn.LayerNorm(self.dim)
        self.mlp = MLP(self.dim, self.dim * mlp_ratio, dropout=drop_rate)

    def forward(self, x):
        x1 = self.ln1(x)
        x = x + self.attn(x1)
        x2 = self.ln2(x)
        x = x + self.mlp(x2)
        return x


class Generator(nn.Module):
    def __init__(self, dim=16, heads=4, mlp_ratio=4, drop_rate=0., device=None, n_class=6, output_dim=11):
        super(Generator, self).__init__()
        self.device = device
        self.dim = dim
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate

        self.positional_embedding = nn.Parameter(torch.randn(1, self.dim)).to(self.device)
        self.encoder = TransformerEncoder(dim=self.dim + n_class, heads=self.heads, mlp_ratio=self.mlp_ratio, drop_rate=self.drop_rate)
        self.linear = nn.Linear(self.dim + n_class, output_dim)

    def forward(self, noise, label):
        patch = ImgPatches().to(self.device)
        x = patch(noise).to(self.device)
        x = x + self.positional_embedding  # batch*1*16
        x = torch.concatenate([x, label], dim=2)
        x = self.encoder(x)
        x = self.linear(x)
        return x.squeeze(1)


class Discriminator(nn.Module):
    def __init__(self, drop_rate=0., dim=11, n_class=6):
        super().__init__()
        self.drop_rate = drop_rate
        self.mlp = MLP(in_feat=dim + n_class, out_feat=1, dropout=self.drop_rate)

    def forward(self, x, label):
        x = torch.concatenate([x, label.squeeze(1)], dim=1)
        return self.mlp(x)

