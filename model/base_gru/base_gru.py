import torch
from torch import nn

from model.base_model import BaseModel
from model.utils import sequence_mask

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

class BaseTransformer(BaseModel):
    def __init__(self, code_num, hidden_dim, max_len):
        super().__init__(param_file_name='base_transformer.pt')
        self.input_proj = nn.Linear(code_num, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim*4, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.positional_encoding = PositionalEncoding(hidden_dim, max_len=max_len)
        self.linear = nn.Sequential(
            nn.Linear(hidden_dim, code_num),
            nn.Sigmoid()
        )
        self.max_len = max_len

    def forward(self, x):
        # x shape: [batch_size, seq_len, code_num]
        x = self.input_proj(x)  # [B, T, H]
        x = self.positional_encoding(x)  # Add position
        out = self.transformer_encoder(x)  # [B, T, H]
        out = self.linear(out)  # [B, T, code_num]
        return out

    def calculate_hidden(self, x, lens):
        with torch.no_grad():
            mask = sequence_mask(lens, self.max_len)  # [B, T]
            x = self.input_proj(x)
            x = self.positional_encoding(x)
            out = self.transformer_encoder(x, src_key_padding_mask=~mask.bool())  # [B, T, H]
            return out * mask.unsqueeze(-1)
