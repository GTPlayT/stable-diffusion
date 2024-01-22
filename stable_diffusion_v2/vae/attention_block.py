from torch.nn import Module, GroupNorm

from stable_diffusion_v2.attention.self_attention import SelfAttention

class AttentionBlock(Module):
    def __init__(self, channels):
        super().__init__()
        self.groupnorm = GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)
    
    def forward(self, x):
        residue = x 

        x = self.groupnorm(x)

        n, c, h, w = x.shape
        x = x.view((n, c, h * w))
        x = x.transpose(-1, -2)
        x = self.attention(x)
        x = x.transpose(-1, -2)
        x = x.view((n, c, h, w))
        x += residue

        return x 
