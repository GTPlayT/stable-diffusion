from torch.nn import ModuleList, LayerNorm, Module
from torch import long, LongTensor, FloatTensor

from stable_diffusion_v2.clip.embedding import EmbeddingCLIP
from stable_diffusion_v2.clip.layer import Layer
from stable_diffusion_v2.clip.variables import EMBEDDING_POSITION, EMBEDDING_DIMENSION, VOCABULARY_SIZE

class CLIP (Module):
    def __init__(self):
        super().__init__()
        self.embedding = EmbeddingCLIP(
            VOCABULARY_SIZE, EMBEDDING_DIMENSION, EMBEDDING_POSITION
            )

        self.layers = ModuleList([
            Layer(12, EMBEDDING_DIMENSION) for i in range(12)
        ])

        self.layernorm = LayerNorm(EMBEDDING_DIMENSION)
    
    def forward(self, tokens: LongTensor) -> FloatTensor:
        tokens = tokens.type(long)
        state = self.embedding(tokens)
        for layer in self.layers: 
            state = layer(state)
        output = self.layernorm(state)
        
        return output