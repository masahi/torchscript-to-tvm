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

pipe.safety_checker = None

# import time
# t1 = time.time()
# sample = pipe("Mt. Fuji in the style of Gauguin", num_inference_steps=1)["images"][0]
# t2 = time.time()

# # # sample.save("out.png")
# # # print(t2 - t1)

class UNetWrapper(torch.nn.Module):
    def __init__(self, model, timestep):
        super().__init__()
        self.model = model
        self.timestep = timestep

    def forward(self, latent_model_input, text_embedding):
        return self.model(latent_model_input, self.timestep, text_embedding).sample


class CLIPWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        return self.model(input_ids)[0]


def serialize(mod, params, prefix):
    with open("{}.json".format(prefix), "w") as fo:
        fo.write(tvm.ir.save_json(mod))
    with open("{}.params".format(prefix), "wb") as fo:
        fo.write(relay.save_param_dict(params))


def deserialize(prefix):
    with open("{}.json".format(prefix), "r") as fi:
        mod = tvm.ir.load_json(fi.read())
    with open("{}.params".format(prefix), "rb") as fi:
        params = relay.load_param_dict(fi.read())
    return mod, params


def export_models():
    timestep = 50

    with torch.no_grad():
        clip_traced = torch.jit.trace(CLIPWrapper(pipe.text_encoder), torch.ones(1, 77, dtype=torch.int64))
        unet_traced = torch.jit.trace(UNetWrapper(pipe.unet, timestep), [torch.randn(2, 4, 64, 64), torch.randn(2, 77, 768)])
        vae_dec_traced = torch.jit.trace(pipe.vae.decoder, torch.randn(1, 4, 64, 64))

    mod_clip, params_clip = relay.frontend.from_pytorch(clip_traced, [("text_input_ids", (1, 77))])
    mod_unet, params_unet = relay.frontend.from_pytorch(unet_traced, [("latent_model_input", (2, 4, 64, 64)), ("text_embedding", (2, 77, 768))])
    mod_vae_dec, params_vae_dec = relay.frontend.from_pytorch(vae_dec_traced, [("latents", (1, 4, 64, 64))])

    serialize(mod_clip, params_clip, "clip")
    serialize(mod_unet, params_unet, "unet")
    serialize(mod_vae_dec, params_vae_dec, "dec")


# export_models()

mod_clip, params_clip = deserialize("clip")
mod_unet, params_unet = deserialize("unet")
mod_dec, params_dec = deserialize("dec")

print(relay.transform.InferType()(mod_unet))
