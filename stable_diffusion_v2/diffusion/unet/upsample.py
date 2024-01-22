from torch.nn import Conv2d, functional, Sequential, Module

from stable_diffusion_v2.diffusion.unet.attention_block import AttentionBlock
from stable_diffusion_v2.diffusion.unet.residual_block import ResidualBlock

class Upsample (Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        x =functional.interpolate(x, scale_factor=2, mode='nearest') 
        return self.conv(x)

class SwitchSequential(Sequential):
    def forward(self, x, context, time):
        for layer in self:
            if isinstance(layer, AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x