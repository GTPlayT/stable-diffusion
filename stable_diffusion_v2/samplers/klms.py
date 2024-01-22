import numpy as np

class Sampler():
    def __init__(self, num_inference_steps=50, n_training_steps=1000, lms_order=4, beta_start=0.00085, beta_end=0.0120, generator=None):
        timesteps = np.linspace(n_training_steps - 1, 0, num_inference_steps)

        betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, n_training_steps, dtype=np.float32) ** 2
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)

        sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        log_sigmas = np.log(sigmas)
        log_sigmas = np.interp(timesteps, range(n_training_steps), log_sigmas)
        sigmas = np.exp(log_sigmas)
        sigmas = np.append(sigmas, 0)
        
        self.sigmas = sigmas
        self.initial_scale = sigmas.max()
        self.timesteps = timesteps
        self.n_inference_steps = num_inference_steps
        self.n_training_steps = n_training_steps
        self.lms_order = lms_order
        self.step_count = 0
        self.outputs = []

    def get_input_scale(self, step_count=None):
        if step_count is None:
            step_count = self.step_count
        sigma = self.sigmas[step_count]
        return 1 / (sigma ** 2 + 1) ** 0.5

    def set_strength(self, strength=1):
        start_step = self.n_inference_steps - int(self.n_inference_steps * strength)
        self.timesteps = np.linspace(self.n_training_steps - 1, 0, self.n_inference_steps)
        self.timesteps = self.timesteps[start_step:]
        self.initial_scale = self.sigmas[start_step]
        self.step_count = start_step

    def step(self, latents, model_output, timestep=None):
        t = self.step_count
        self.step_count += 1

        self.outputs = [ model_output] + self.outputs[:self.lms_order - 1]
        order = len(self.outputs)
        for i, output in enumerate(self.outputs):
            x = np.linspace(self.sigmas[t], self.sigmas[t + 1], 81)
            y = np.ones(81)
            for j in range(order):
                if i == j:
                    continue
                y *= x - self.sigmas[t - j]
                y /= self.sigmas[t - i] - self.sigmas[t - j]
            lms_coeff = np.trapz(y=y, x=x)
            latents += lms_coeff * output
        return latents