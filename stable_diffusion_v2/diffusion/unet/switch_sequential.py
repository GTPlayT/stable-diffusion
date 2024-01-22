from torch.nn import Sequential

from stable_diffusion_v2.diffusion.unet.attention_block import AttentionBlock
from stable_diffusion_v2.diffusion.unet.residual_block import ResidualBlock

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