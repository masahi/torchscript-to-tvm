import time

from diffusers import StableDiffusionPipeline
import torch

import tvm
from tvm import relay


pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    # revision="fp16",
    # torch_dtype=torch.float16,
)

# import time
# t1 = time.time()
# sample = pipe("Mt. Fuji in the style of Gauguin", num_inference_steps=1)["images"][0]
# t2 = time.time()

# sample.save("out.png")
# print(t2 - t1)

class UNetWrapper(torch.nn.Module):
    def __init__(self, model, timestep):
        super().__init__()
        self.model = model
        self.timestep = timestep

    def forward(self, latent_model_input, text_embedding):
        return self.model(latent_model_input, self.timestep, text_embedding).sample

timestep = 50

with torch.no_grad():
    unet_traced = torch.jit.trace(UNetWrapper(pipe.unet, timestep), [torch.randn(2, 4, 64, 64), torch.randn(2, 77, 768)])
    vae_dec_traced = torch.jit.trace(pipe.vae.decoder, torch.randn(1, 4, 64, 64))

t1 = time.time()
mod_unet, params_unet = relay.frontend.from_pytorch(unet_traced, [("latent_model_input", (2, 4, 64, 64)), ("text_embedding", (2, 77, 768))])
t2 = time.time()

print(relay.transform.InferType()(mod_unet))
print("UNet converted in ", t2 - t1)

mod_vae_dec, params_vae_dec = relay.frontend.from_pytorch(vae_dec_traced, [("latents", (1, 4, 64, 64))])

# print(relay.transform.InferType()(mod_vae_dec))
