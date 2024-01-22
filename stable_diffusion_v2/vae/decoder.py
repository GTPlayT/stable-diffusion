from torch.nn import Sequential, Conv2d, Upsample, GroupNorm, SiLU

from stable_diffusion_v2.vae.residual_block import ResidualBlock
from stable_diffusion_v2.vae.attention_block import AttentionBlock

class Decoder (Sequential):
    def __init__(self):
        super().__init__(
            Conv2d(4, 4, kernel_size=1, padding=0),

            Conv2d(4, 512, kernel_size=3, padding=1),
            ResidualBlock(512, 512), 
            AttentionBlock(512), 
            ResidualBlock(512, 512), 
            ResidualBlock(512, 512), 
            ResidualBlock(512, 512), 
            ResidualBlock(512, 512), 
            Upsample(scale_factor=2),

            Conv2d(512, 512, kernel_size=3, padding=1), 
            ResidualBlock(512, 512), 
            ResidualBlock(512, 512), 
            ResidualBlock(512, 512), 
            Upsample(scale_factor=2), 

            Conv2d(512, 512, kernel_size=3, padding=1), 
            ResidualBlock(512, 256), 
            ResidualBlock(256, 256), 
            ResidualBlock(256, 256), 
            Upsample(scale_factor=2), 

            Conv2d(256, 256, kernel_size=3, padding=1), 
            ResidualBlock(256, 128), 
            ResidualBlock(128, 128), 
            ResidualBlock(128, 128), 
            
            GroupNorm(32, 128), 
            SiLU(), 
            
            Conv2d(128, 3, kernel_size=3, padding=1), 
        )

    def forward(self, x):
        x /= 0.18215

        for module in self:
            x = module(x)

        return x