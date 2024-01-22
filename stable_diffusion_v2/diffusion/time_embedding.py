from torch.nn import Module, Linear, functional

class TimeEmbedding (Module):
    def __init__(self, n_embd):
        super().__init__()
        self.linear_1 = Linear(n_embd, 4 * n_embd)
        self.linear_2 = Linear(4 * n_embd, 4 * n_embd)

    def forward(self, x):
        x = self.linear_1(x)
        x = functional.silu(x) 
        x = self.linear_2(x)

        return x