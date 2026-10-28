"""HELENA-CE model components."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, mask, adjoin_matrix, dist_matrix):
        batch_size = q.size(0)

        q = self.split_heads(self.Wq(q), batch_size)
        k = self.split_heads(self.Wk(k), batch_size)
        v = self.split_heads(self.Wv(v), batch_size)

        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask, adjoin_matrix, dist_matrix
        )

        scaled_attention = scaled_attention.permute(0, 2, 1, 3)
        concat_attention = scaled_attention.reshape(batch_size, -1, self.d_model)
        output = self.dense(concat_attention)

        return output, attention_weights


class EncoderLayer(nn.Module):
    """Dual-path encoder: local (adjacency) + global (distance) attention."""

    def __init__(self, d_model, num_heads, dff, rate):
        super().__init__()
        self.multi_head_att_1 = MultiHeadAttention(d_model // 2, num_heads)
        self.multi_head_att_2 = MultiHeadAttention(d_model // 2, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff), nn.GELU(), nn.Linear(dff, d_model)
        )
        self.layer_norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, x, encoder_padding_mask, adjoin_matrix, dist_matrix):
        x1, x2 = torch.chunk(x, 2, dim=-1)

        x_l, attn_weights_local = self.multi_head_att_1(
            x1, x1, x1, encoder_padding_mask, adjoin_matrix, dist_matrix=None
        )
        x_g, attn_weights_global = self.multi_head_att_2(
            x2, x2, x2, encoder_padding_mask, adjoin_matrix=None, dist_matrix=dist_matrix
        )

        attn_output = torch.cat([x_l, x_g], dim=-1)
        out1 = self.layer_norm1(x + self.dropout1(attn_output))

        ffn_output = self.ffn(out1)
        out2 = self.layer_norm2(out1 + self.dropout2(ffn_output))

        return out2, attn_weights_local, attn_weights_global


class MoietyEncoder(nn.Module):
    def __init__(self, input_vocab_size, d_model, dff, num_layers, num_heads, rate):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_vocab_size, d_model)
        self.dropout = nn.Dropout(rate)
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        )
        self.moiety_linear = nn.Linear(d_model, d_model)

    def forward(self, x, adjoin_matrix=None, dist_matrix=None):
        padding_mask = create_padding_mask(x)

        if adjoin_matrix is not None:
            adjoin_matrix = adjoin_matrix.unsqueeze(1)
        if dist_matrix is not None:
            dist_matrix = dist_matrix.unsqueeze(1)

        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.moiety_linear(self.dropout(x))

        attn_weights_local_list = []
        attn_weights_global_list = []

        for layer in self.encoder_layers:
            x, attn_local, attn_global = layer(x, padding_mask, adjoin_matrix, dist_matrix)
            attn_weights_local_list.append(attn_local)
            attn_weights_global_list.append(attn_global)

        return x, attn_weights_local_list, attn_weights_global_list, padding_mask


class MolarRatioEncoder(nn.Module):
    """Normalizes molar ratios to weights for molecular embeddings."""

    def __init__(self, d_model, hidden_dim=64):
        super().__init__()
        self.d_model = d_model

    def forward(self, molar_ratios):
        zero_mask = molar_ratios == 0
        masked_ratios = molar_ratios.masked_fill(zero_mask, 0.0)
        ratio_sum = masked_ratios.sum(dim=1, keepdim=True).clamp(min=1e-6)
        mr_weights = (masked_ratios / ratio_sum).unsqueeze(-1)
        return mr_weights.expand(-1, -1, self.d_model)


class FormulationAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_mask = ~mask
        empty_samples = ~mask.any(dim=1)
        if empty_samples.any():
            attn_mask = attn_mask.clone()
            attn_mask[empty_samples, 0] = False

        attended_x, attention_weights = self.attention(
            query=x, key=x, value=x, key_padding_mask=attn_mask
        )
        x = self.layer_norm(x + self.dropout(attended_x))
        x = torch.nan_to_num(x * mask.unsqueeze(-1).float(), nan=0.0)
        return x, attention_weights


class GlobalPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask):
        masked_x = x * mask.unsqueeze(-1)
        return torch.nan_to_num(masked_x.sum(dim=1), nan=0.0)


class SolventGroup(nn.Module):
    """Multi-layer self-attention for solvent molecules with mean pooling."""

    def __init__(self, d_model, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])

    def forward(self, scaled_mol_embeddings, mask):
        attn_mask = ~mask
        x = scaled_mol_embeddings

        empty_samples = ~mask.any(dim=1)
        if empty_samples.any():
            attn_mask = attn_mask.clone()
            attn_mask[empty_samples, 0] = False

        for attn, norm in zip(self.layers, self.norms):
            out, weights = attn(x, x, x, key_padding_mask=attn_mask)
            x = norm(out + x)

        valid_count = mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)
        pooled = (x * mask.unsqueeze(-1).float()).sum(dim=1) / valid_count
        return torch.nan_to_num(pooled, nan=0.0), weights


class SaltGroup(nn.Module):
    """Mean pooling for salt molecules."""

    def __init__(self):
        super().__init__()

    def forward(self, mol_embeddings, mask):
        masked_embeddings = mol_embeddings * mask.unsqueeze(-1).float()
        valid_count = mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)
        pooled = masked_embeddings.sum(dim=1) / valid_count
        return torch.nan_to_num(pooled, nan=0.0)


def create_padding_mask(batch_data):
    return (batch_data == 0).float().unsqueeze(1).unsqueeze(2)


def rescale_distance_matrix(w):
    c = torch.tensor(1.0, dtype=torch.float32, device=w.device)
    return (c + torch.exp(c)) / (c + torch.exp(c - w))


def scaled_dot_product_attention(q, k, v, mask, adjoin_matrix, dist_matrix):
    dk = q.size(-1)
    matmul_qk = torch.matmul(q, k.transpose(-2, -1))

    if dist_matrix is not None:
        matmul_qk = F.relu(matmul_qk)
        dist_matrix = rescale_distance_matrix(dist_matrix)
        scaled_attention_logits = (matmul_qk * dist_matrix) / math.sqrt(dk)
    elif adjoin_matrix is not None:
        scaled_attention_logits = (matmul_qk * adjoin_matrix) / math.sqrt(dk)
    else:
        scaled_attention_logits = matmul_qk / math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits = scaled_attention_logits + mask * -1e9

    attention_weights = F.softmax(scaled_attention_logits, dim=-1)
    output = torch.matmul(attention_weights, v)

    return output, attention_weights
