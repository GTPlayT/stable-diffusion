from torch.nn import Module, Conv2d, ModuleList
from torch import cat

from stable_diffusion_v2.diffusion.unet.attention_block import AttentionBlock
from stable_diffusion_v2.diffusion.unet.residual_block import ResidualBlock
from stable_diffusion_v2.diffusion.unet.switch_sequential import SwitchSequential
from stable_diffusion_v2.diffusion.unet.upsample import Upsample

class UNET (Module):
    def __init__(self):
        super().__init__()
        self.encoders = ModuleList([
            SwitchSequential(Conv2d(4, 320, kernel_size=3, padding=1)),
            SwitchSequential(ResidualBlock(320, 320), AttentionBlock(8, 40)),
            SwitchSequential(ResidualBlock(320, 320), AttentionBlock(8, 40)),

            SwitchSequential(Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(ResidualBlock(320, 640), AttentionBlock(8, 80)),
            SwitchSequential(ResidualBlock(640, 640), AttentionBlock(8, 80)),
            
            SwitchSequential(Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(ResidualBlock(640, 1280), AttentionBlock(8, 160)),
            SwitchSequential(ResidualBlock(1280, 1280), AttentionBlock(8, 160)),
            
            SwitchSequential(Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(ResidualBlock(1280, 1280)),
            SwitchSequential(ResidualBlock(1280, 1280)),
        ])

        self.bottleneck = SwitchSequential(
            ResidualBlock(1280, 1280), 
            AttentionBlock(8, 160), 
            ResidualBlock(1280, 1280), 
        )
        
        self.decoders = ModuleList([
            SwitchSequential(ResidualBlock(2560, 1280)),
            SwitchSequential(ResidualBlock(2560, 1280)),
            SwitchSequential(ResidualBlock(2560, 1280), Upsample(1280)),
            SwitchSequential(ResidualBlock(2560, 1280), AttentionBlock(8, 160)),
            SwitchSequential(ResidualBlock(2560, 1280), AttentionBlock(8, 160)),
            SwitchSequential(ResidualBlock(1920, 1280), AttentionBlock(8, 160), Upsample(1280)),
            SwitchSequential(ResidualBlock(1920, 640), AttentionBlock(8, 80)),
            SwitchSequential(ResidualBlock(1280, 640), AttentionBlock(8, 80)),
            SwitchSequential(ResidualBlock(960, 640), AttentionBlock(8, 80), Upsample(640)),
            SwitchSequential(ResidualBlock(960, 320), AttentionBlock(8, 40)),
            SwitchSequential(ResidualBlock(640, 320), AttentionBlock(8, 40)),
            SwitchSequential(ResidualBlock(640, 320), AttentionBlock(8, 40)),
        ])

    def forward(self, x, context, time):
        skip_connections = []
        for layers in self.encoders:
            x = layers(x, context, time)
            skip_connections.append(x)

        x = self.bottleneck(x, context, time)

        for layers in self.decoders:
            x = cat((x, skip_connections.pop()), dim=1) 
            x = layers(x, context, time)
        
        return x
