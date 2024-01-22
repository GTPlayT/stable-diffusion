from torch.nn import Module, GroupNorm, Conv2d, functional

class OutputLayer (Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm = GroupNorm(32, in_channels)
        self.conv = Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        
        x = self.groupnorm(x)
        x = functional.silu(x)
        x = self.conv(x)

        return x