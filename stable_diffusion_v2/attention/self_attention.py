from torch import ones_like, bool, inf
from torch.nn import Module, Linear, functional

from math import sqrt

class SelfAttention (Module):
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        self.in_proj = Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = Linear(d_embed, d_embed, bias=out_proj_bias)
        
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
    
    def forward (self, x, causal_mask=False):
        input_shape = x.shape

        batch_size, sequence_len, d_embed = input_shape
        interim_shape = (batch_size, sequence_len, self.n_heads, self.d_head)

        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2)

        if causal_mask:
            mask = ones_like(weight, dtype=bool).triu(1)
            weight.masked_fill_(mask, -inf)

        weight /= sqrt(self.d_head)
        weight = functional.softmax(weight, dim=-1)

        output = weight @ v
        output = output.transpose(1, 2)
        output = output.reshape(input_shape)
        output = self.out_proj(output)

        return output