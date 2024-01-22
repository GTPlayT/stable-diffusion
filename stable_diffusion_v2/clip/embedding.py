from torch.nn import Embedding, Parameter, Module
from torch import zeros

class EmbeddingCLIP (Module):
    def __init__(self, n_vocab: int, n_embed: int, n_token: int) -> None:
        super().__init__()

        self.token_embedding = Embedding(n_vocab, n_embed)
        self.position_embedding = Parameter(zeros((n_token, n_embed)))

    def forward (self, tokens):
        x = self.token_embedding(tokens)
        x += self.position_embedding

        return x