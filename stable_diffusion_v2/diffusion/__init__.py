from torch.nn import Module

from stable_diffusion_v2.diffusion.unet.unet import UNET
from stable_diffusion_v2.diffusion.time_embedding import TimeEmbedding
from stable_diffusion_v2.diffusion.unet import OutputLayer

class Diffusion(Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = OutputLayer(320, 4)
    
    def forward(self, latent, context, time):
        time = self.time_embedding(time)
        output = self.unet(latent, context, time)
        output = self.final(output)
        
        return output