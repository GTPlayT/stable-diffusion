def load_sampler (name: str = "ddpm"):
    if name == "ddpm":
        from stable_diffusion_v2.samplers.ddpm import Sampler
        return Sampler
    elif name == "ddim":
        from stable_diffusion_v2.samplers.ddim import Sampler
        return Sampler
    elif name == "klms":
        from stable_diffusion_v2.samplers.klms import Sampler
        return Sampler
    elif name == "kea":
        from stable_diffusion_v2.samplers.kea import Sampler
        return Sampler
    elif name == "keuler":
        from stable_diffusion_v2.samplers.keuler import Sampler
        return Sampler
    else:
        raise ValueError("Please give a proper sampler name!")