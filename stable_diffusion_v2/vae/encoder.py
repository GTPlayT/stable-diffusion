from torch.nn import Sequential, Conv2d, Upsample, GroupNorm, SiLU, functional
from torch import clamp, chunk

from stable_diffusion_v2.vae.residual_block import ResidualBlock
from stable_diffusion_v2.vae.attention_block import AttentionBlock

class Encoder(Sequential):
    def __init__(self):
        super().__init__(
            Conv2d(3, 128, kernel_size=3, padding=1),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            
            Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            ResidualBlock(128, 256), 
            ResidualBlock(256, 256), 

            Conv2d(256, 256, kernel_size=3, stride=2, padding=0), 
            ResidualBlock(256, 512), 
            ResidualBlock(512, 512), 
            
            Conv2d(512, 512, kernel_size=3, stride=2, padding=0), 
            ResidualBlock(512, 512), 
            ResidualBlock(512, 512), 
            ResidualBlock(512, 512), 
            AttentionBlock(512), 
            ResidualBlock(512, 512), 
            
            GroupNorm(32, 512),
            SiLU(), 

            Conv2d(512, 8, kernel_size=3, padding=1), 
            Conv2d(8, 8, kernel_size=1, padding=0), 
        )

    def forward(self, x, noise):
        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                x = functional.pad(x, (0, 1, 0, 1))
            
            x = module(x)
        mean, log_variance = chunk(x, 2, dim=1)
        log_variance = clamp(log_variance, -30, 20)
        variance = log_variance.exp()

        stdev = variance.sqrt()
        x = mean + stdev * noise
        x *= 0.18215
        
        return x
