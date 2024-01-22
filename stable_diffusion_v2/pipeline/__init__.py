from torch import no_grad, Generator, long, tensor, cat, randn, float32, uint8
from numpy import array
from tqdm import tqdm

from stable_diffusion_v2.samplers import load_sampler
from stable_diffusion_v2.pipeline.rescale import rescale
from stable_diffusion_v2.pipeline.get_time_embedding import get_time_embedding

def generate(
    prompt,
    uncond_prompt=None,
    height = 512,
    width = 512,
    input_image=None,
    strength=0.8,
    do_cfg=True,
    cfg_scale=7.5,
    sampler_name="ddpm",
    n_inference_steps=50,
    models={},
    seed=None,
    device=None,
    idle_device=None,
    tokenizer=None,
):
    
    with no_grad():
        if not 0 < strength <= 1:
            raise ValueError("strength must be between 0 and 1")

        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x
        generator = Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        clip = models["clip"]
        clip.to(device)
        
        if do_cfg:
            cond_tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            cond_tokens = tensor(cond_tokens, dtype=long, device=device)
            cond_context = clip(cond_tokens)
            uncond_tokens = tokenizer.batch_encode_plus(
                [uncond_prompt], padding="max_length", max_length=77
            ).input_ids
            uncond_tokens = tensor(uncond_tokens, dtype=long, device=device)
            uncond_context = clip(uncond_tokens)
            context = cat([cond_context, uncond_context])
        else:
            tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            tokens = tensor(tokens, dtype=long, device=device)
            context = clip(tokens)
        to_idle(clip)

        sampler = load_sampler(sampler_name)
        sampler = sampler(generator=generator, num_inference_steps=n_inference_steps)

        latents_height = height // 8
        latents_width = width // 8

        latents_shape = (1, 4, latents_height, latents_width)

        if input_image:
            encoder = models["encoder"]
            encoder.to(device)

            input_image_tensor = input_image.resize((width, height))
            input_image_tensor = array(input_image_tensor)
            input_image_tensor = tensor(input_image_tensor, dtype=float32)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            input_image_tensor = input_image_tensor.unsqueeze(0)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            encoder_noise = randn(latents_shape, generator=generator, device=device)
            latents = encoder(input_image_tensor, encoder_noise)

            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(encoder)
        else:
            latents = randn(latents_shape, generator=generator, device=device)

        diffusion = models["diffusion"]
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            time_embedding = get_time_embedding(timestep).to(device)
            model_input = latents

            if do_cfg:
                model_input = model_input.repeat(2, 1, 1, 1)
            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond
            latents = sampler.step(timestep = timestep, latents = latents, model_output=model_output)

        to_idle(diffusion)

        decoder = models["decoder"]
        images = decoder(latents)
        to_idle(decoder)

        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", uint8).numpy()
        return images[0]

