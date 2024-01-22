from torch.nn import Module, GroupNorm, Conv2d, Identity, functional

class ResidualBlock (Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = GroupNorm(32, in_channels)
        self.conv_1 = Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = GroupNorm(32, out_channels)
        self.conv_2 = Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = Identity()
        else:
            self.residual_layer = Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, x):
        residue = x

        x = self.groupnorm_1(x)
        x = functional.silu(x)
        x = self.conv_1(x)
        x = self.groupnorm_2(x)
        x = functional.silu(x)
        x = self.conv_2(x)
        return x + self.residual_layer(residue)