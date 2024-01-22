from torch.nn import Module, LayerNorm, Linear
from torch import sigmoid

from stable_diffusion_v2.attention.self_attention import SelfAttention

class Layer (Module):
    def __init__(self, n_head: int, n_embed: int) -> None:
        super().__init__()

        self.layernorm_1 = LayerNorm(n_embed)
        self.attention = SelfAttention(n_head, n_embed)

        self.layernorm_2 = LayerNorm(n_embed)

        self.linear_1 = Linear(n_embed, 4 * n_embed)
        self.linear_2 = Linear(4 * n_embed, n_embed)

    def forward(self, x):
        residue = x
        
        x = self.layernorm_1(x)
        x = self.attention(x, causal_mask=True)
        x += residue

        residue = x

        x = self.layernorm_2(x)
        x = self.linear_1(x)
        x = x * sigmoid(1.702 * x)   
        x = self.linear_2(x)
        x += residue

        return x