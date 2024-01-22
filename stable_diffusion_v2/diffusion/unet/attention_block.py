from torch.nn import GroupNorm, Conv2d, LayerNorm, Module, functional, Linear

from stable_diffusion_v2.attention.self_attention import SelfAttention
from stable_diffusion_v2.attention.cross_attention import CrossAttention

class AttentionBlock (Module):
    def __init__(self, n_head: int, n_embd: int, d_context=768):
        super().__init__()
        channels = n_head * n_embd
        
        self.groupnorm = GroupNorm(32, channels, eps=1e-6)
        self.conv_input = Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1 = LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm_2 = LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        self.layernorm_3 = LayerNorm(channels)
        self.linear_geglu_1  = Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = Linear(4 * channels, channels)

        self.conv_output = Conv2d(channels, channels, kernel_size=1, padding=0)
    
    def forward(self, x, context):
        residue_long = x

        x = self.groupnorm(x)
        x = self.conv_input(x)
        
        n, c, h, w = x.shape
        
        x = x.view((n, c, h * w))
        x = x.transpose(-1, -2)

        residue_short = x

        x = self.layernorm_1(x)
        x = self.attention_1(x)
        x += residue_short

        residue_short = x

        x = self.layernorm_2(x)
        x = self.attention_2(x, context)
        x += residue_short

        residue_short = x

        x = self.layernorm_3(x)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1) 
        x = x *functional.gelu(gate)
        x = self.linear_geglu_2(x)
        x += residue_short
        x = x.transpose(-1, -2)
        x = x.view((n, c, h, w))

        return self.conv_output(x) + residue_long