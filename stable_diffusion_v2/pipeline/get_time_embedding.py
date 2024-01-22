from torch import arange, tensor, cat, cos, sin, float32

def get_time_embedding(timestep):
    freqs = pow(10000, -arange(start=0, end=160, dtype=float32) / 160) 
    x = tensor([timestep], dtype=float32)[:, None] * freqs[None]
    return cat([cos(x), sin(x)], dim=-1)