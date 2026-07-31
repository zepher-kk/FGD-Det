from __future__ import annotations
"""
CTF: Cross-Transformer-based fusion modules for image-image (图-图) multimodal tasks.

导出类（对外公开使用）：
- CrossTransformerFusion：用于两路 [B,C,H,W] 特征的跨模态 Transformer 编码与融合，输出 [B,2C,H,W]
- MultiHeadCrossAttention：底层多头交叉注意力（序列级 [B,N,C] 输入），供高级/自定义编排使用
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class MultiHeadCrossAttention(nn.Module):









































    def __init__(self, model_dim, num_heads):
        super(MultiHeadCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        assert (self.head_dim * num_heads == model_dim), "model_dim must be divisible by num_heads"


        self.query_vis = nn.Linear(model_dim, model_dim)
        self.key_vis = nn.Linear(model_dim, model_dim)
        self.value_vis = nn.Linear(model_dim, model_dim)


        self.query_inf = nn.Linear(model_dim, model_dim)
        self.key_inf = nn.Linear(model_dim, model_dim)
        self.value_inf = nn.Linear(model_dim, model_dim)


        self.fc_out_vis = nn.Linear(model_dim, model_dim)
        self.fc_out_inf = nn.Linear(model_dim, model_dim)

    def forward(self, vis, inf):
        batch_size, seq_length, model_dim = vis.shape


        Q_vis = self.query_vis(vis)
        K_vis = self.key_vis(vis)
        V_vis = self.value_vis(vis)


        Q_inf = self.query_inf(inf)
        K_inf = self.key_inf(inf)
        V_inf = self.value_inf(inf)


        def reshape_heads(x):
            return x.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        Q_vis = reshape_heads(Q_vis)
        K_vis = reshape_heads(K_vis)
        V_vis = reshape_heads(V_vis)
        Q_inf = reshape_heads(Q_inf)
        K_inf = reshape_heads(K_inf)
        V_inf = reshape_heads(V_inf)


        scale = (self.head_dim ** -0.5)
        scores_vis_inf = torch.matmul(Q_vis, K_inf.transpose(-1, -2)) * scale
        scores_inf_vis = torch.matmul(Q_inf, K_vis.transpose(-1, -2)) * scale

        attn_inf = torch.softmax(scores_vis_inf, dim=-1)
        attn_vis = torch.softmax(scores_inf_vis, dim=-1)

        out_inf = torch.matmul(attn_inf, V_inf)
        out_vis = torch.matmul(attn_vis, V_vis)


        out_vis = out_vis.transpose(1, 2).contiguous().view(batch_size, seq_length, model_dim)
        out_inf = out_inf.transpose(1, 2).contiguous().view(batch_size, seq_length, model_dim)

        out_vis = self.fc_out_vis(out_vis)
        out_inf = self.fc_out_inf(out_inf)
        return out_vis, out_inf

class FeedForward(nn.Module):








































    def __init__(self, model_dim, hidden_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(model_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class PositionalEncoding(nn.Module):









































    def __init__(self, model_dim, dropout, max_len=6400):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2) * -(torch.log(torch.tensor(10000.0)) / model_dim))
        pe = torch.zeros(max_len, model_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerEncoderLayer(nn.Module):








































    def __init__(self, model_dim, num_heads, hidden_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.cross_attention = MultiHeadCrossAttention(model_dim, num_heads)
        self.norm1 = nn.LayerNorm(model_dim)
        self.ff = FeedForward(model_dim, hidden_dim, dropout)
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, vis, inf):
        attn_out_vis, attn_out_inf = self.cross_attention(vis, inf)
        vis = self.norm1(vis + attn_out_vis)
        inf = self.norm1(inf + attn_out_inf)
        ff_out_vis = self.ff(vis)
        ff_out_inf = self.ff(inf)
        vis = self.norm2(vis + ff_out_vis)
        inf = self.norm2(inf + ff_out_inf)
        return vis, inf

class TransformerEncoder(nn.Module):










































    def __init__(self, input_dim, model_dim, num_heads, num_layers, hidden_dim, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim, dropout)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(model_dim, num_heads, hidden_dim, dropout) for _ in range(num_layers)
        ])

    def forward(self, vis, inf):
        vis = self.embedding(vis) * torch.sqrt(torch.tensor(self.embedding.out_features, dtype=torch.float32))
        inf = self.embedding(inf) * torch.sqrt(torch.tensor(self.embedding.out_features, dtype=torch.float32))
        vis = self.positional_encoding(vis)
        inf = self.positional_encoding(inf)
        for layer in self.layers:
            vis, inf = layer(vis, inf)
        return vis, inf

class CrossTransformerFusion(nn.Module):













































    def __init__(self, input_dim, num_heads=2, num_layers=1, dropout=0.1):
        super(CrossTransformerFusion, self).__init__()
        self.hidden_dim = input_dim * 2
        self.model_dim = input_dim
        self.encoder = TransformerEncoder(input_dim, self.model_dim, num_heads, num_layers, self.hidden_dim, dropout)

    def forward(self, x):
        vis, inf = x[0], x[1]
        B, C, H, W = vis.shape
        vis = vis.permute(0, 2, 3, 1).reshape(B, -1, C)
        inf = inf.permute(0, 2, 3, 1).reshape(B, -1, C)
        vis_out, inf_out = self.encoder(vis, inf)
        vis_out = vis_out.view(B, H, W, -1).permute(0, 3, 1, 2)
        inf_out = inf_out.view(B, H, W, -1).permute(0, 3, 1, 2)
        out = torch.cat((vis_out, inf_out), dim=1)
        return out

