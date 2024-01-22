from torch.nn import Module, GroupNorm, Conv2d, Linear, Identity, functional

class ResidualBlock(Module):
    def __init__(self, in_channels, out_channels, n_time=1280):
        super().__init__()
        self.groupnorm_feature = GroupNorm(32, in_channels)
        self.conv_feature = Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = Linear(n_time, out_channels)

        self.groupnorm_merged = GroupNorm(32, out_channels)
        self.conv_merged = Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = Identity()
        else:
            self.residual_layer = Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, feature, time):
        residue = feature

        feature = self.groupnorm_feature(feature)
        feature = functional.silu(feature)
        feature = self.conv_feature(feature)

        time =functional.silu(time)
        time = self.linear_time(time)
        
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        merged = self.groupnorm_merged(merged)
        merged =functional.silu(merged)
        merged = self.conv_merged(merged)
        
        return merged + self.residual_layer(residue)