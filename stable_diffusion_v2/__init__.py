from torch import cuda
from transformers import CLIPTokenizer
from PIL.Image import fromarray
from json import load

from stable_diffusion_v2 import model_converter
from stable_diffusion_v2.vae.encoder import Encoder
from stable_diffusion_v2.vae.decoder import Decoder
from stable_diffusion_v2.clip import CLIP
from stable_diffusion_v2.diffusion import Diffusion
from stable_diffusion_v2.pipeline import generate

def preload_models_from_standard_weights(model_path, device):
    state_dict = model_converter.load_from_standard_weights(model_path, device)

    encoder = Encoder().to(device)
    encoder.load_state_dict(state_dict['encoder'], strict=True)

    decoder = Decoder().to(device)
    decoder.load_state_dict(state_dict['decoder'], strict=True)

    diffusion = Diffusion().to(device)
    diffusion.load_state_dict(state_dict['diffusion'], strict=True)

    clip = CLIP().to(device)
    clip.load_state_dict(state_dict['clip'], strict=True)

    return {
        'clip': clip,
        'encoder': encoder,
        'decoder': decoder,
        'diffusion': diffusion,
    }

class StableDiffusion:
    def __init__ (self, model="", sampler_name="ddpm", device="cpu"):
        self.sampler_name = sampler_name

        self.device = device
        if cuda.is_available():
            self.device = "cuda"

        file = open("stable_diffusion_v2/data/models/models.json", "r")
        model_location = load(file)

        if model in model_location:
            self.model = preload_models_from_standard_weights(model_path=model_location[model], device=self.device)
        else:
            raise ValueError("Couldn't find the required model.")

        self.tokenizer = CLIPTokenizer("stable_diffusion_v2/data/text/vocab.json", merges_file="stable_diffusion_v2/data/text/merges.txt")

    
    def generate(
        self,
        prompt,
        uncond_prompt="",
        height = 512,
        width = 512,
        input_image=None,
        strength=0.8,
        do_cfg=True,
        cfg_scale=7.5,
        steps=30,
        seed=None,
    ):

        output = generate(
            prompt=prompt,
            uncond_prompt=uncond_prompt,
            input_image=input_image,
            strength=strength,
            do_cfg=do_cfg,
            cfg_scale=cfg_scale,
            sampler_name=self.sampler_name,
            n_inference_steps=steps,
            seed=seed,
            models=self.model,
            device=self.device,
            idle_device="cpu",
            tokenizer=self.tokenizer,
            height = height,
            width = width,
        )

        return fromarray(output)
        
