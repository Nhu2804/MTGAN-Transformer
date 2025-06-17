import torch
from torch import nn

from model.utils import MaskedAttention
import math


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_len):
        super().__init__()
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # [1, max_len, hidden_dim]

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)


class TransformerGenerator(nn.Module):
    def __init__(self, code_num, hidden_dim, max_len, device=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.device = device

        self.noise2input = nn.Linear(code_num, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 4, dropout=0.1, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, code_num),
            nn.Sigmoid()
        )

    def forward(self, noise, target_codes=None):
        # noise: [B, code_num]
        batch_size = noise.size(0)

        input_seq = noise.unsqueeze(1).repeat(1, self.max_len, 1)  # [B, T, code_num]
        input_seq = self.noise2input(input_seq)  # [B, T, H]

        # ✳️ Nhúng mã bệnh mục tiêu vào từng bước
        if target_codes is not None:
            # one-hot: [B, code_num]
            target_embed = torch.zeros((batch_size, self.code_num), device=noise.device)
            target_embed[torch.arange(batch_size), target_codes] = 1
            target_embed = self.noise2input(target_embed).unsqueeze(1)  # [B, 1, H]
            target_embed = target_embed.repeat(1, self.max_len, 1)
            input_seq = input_seq + target_embed  # ép điều kiện

        input_seq = self.pos_encoder(input_seq)
        hiddens = self.transformer_encoder(input_seq)
        samples = self.output_layer(hiddens)
        return samples, hiddens



class SmoothCondition(nn.Module):
    def __init__(self, code_num, attention_dim):
        super().__init__()
        self.attention = MaskedAttention(code_num, attention_dim)

    def forward(self, x, lens, target_codes):
        score = self.attention(x, lens)
        score_tensor = torch.zeros_like(x)
        score_tensor[torch.arange(len(x)), :, target_codes] = score
        x = x + score_tensor
        x = torch.clip(x, max=1)
        return x
