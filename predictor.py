import torch
import torch.nn as nn
from encoder import MLP
class FFN(nn.Module):
    def __init__(self, hidden_dim = 512):
        super().__init__()
        self.layer1 = nn.Linear(192, hidden_dim) 
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, 192)   
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


class AdaLN(nn.Module):
    def __init__(self, a_dim, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim, elementwise_affine=False)
        self.gamma = nn.Linear(a_dim, dim)
        self.beta = nn.Linear(a_dim, dim)
        nn.init.zeros_(self.gamma.weight); nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight); nn.init.zeros_(self.beta.bias)

    def forward(self, x, a):
        gamma = self.gamma(a)
        beta = self.beta(a)
        x = self.ln(x)
        x = gamma * x + beta
        return x

class DecoderBlock(nn.Module):
    def __init__(self,dim=192, n_head=16, a_dim=2):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_head, batch_first=True, dropout=0.1)
        self.adaln_attn = AdaLN(a_dim=a_dim, dim=dim)
        self.adaln_ffn = AdaLN(a_dim=a_dim, dim=dim)
        self.ffn = FFN(hidden_dim=dim)

        # optional residual gates
        self.gate_attn = nn.Linear(a_dim, dim)
        self.gate_ffn = nn.Linear(a_dim, dim)
        nn.init.zeros_(self.gate_attn.weight); nn.init.zeros_(self.gate_attn.bias)
        nn.init.zeros_(self.gate_ffn.weight); nn.init.zeros_(self.gate_ffn.bias)

    def forward(self, x, a, attn_mask=None):
        # x: B, N, D
        # a: B, N, A
        h = self.adaln_attn(x, a)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask) # self attention, skip masking for now
                               # since it N =1
        x = x + torch.tanh(self.gate_attn(a)) * h      
        h = self.adaln_ffn(x, a)
        x = x + torch.tanh(self.gate_ffn(a)) * self.ffn(h)

        return x

class Predictor(nn.Module):
    def __init__(self, a_dim=2, dim=192):
       super().__init__()
       self.transformer_stack = nn.ModuleList([
           DecoderBlock(a_dim=a_dim, dim=dim),
           DecoderBlock(a_dim=a_dim, dim=dim),
           DecoderBlock(a_dim=a_dim, dim=dim),
           DecoderBlock(a_dim=a_dim, dim=dim),
           DecoderBlock(a_dim=a_dim, dim=dim),
           DecoderBlock(a_dim=a_dim, dim=dim)
       ])
       self.mlp = MLP(input_size=dim, output_size=dim)


    def forward(self, z_t, a_t):
        # z_t: B, N, D (B, 1, 192) N=1 for tworooms as per paper.  
        # a_t: B, N, A (B, 1, 2)
        x = z_t
        for block in self.transformer_stack:
            x = block(x, a_t)
        x = self.mlp(x)
        # there is only one token in sequence so just return it as the last one
        return x