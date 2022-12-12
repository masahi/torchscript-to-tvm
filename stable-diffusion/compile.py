import time
from collections import namedtuple

from diffusers import StableDiffusionPipeline
import torch

import tvm
from tvm import relay


def deserialize(prefix):
    with open("{}.json".format(prefix), "r") as fi:
        mod = tvm.ir.load_json(fi.read())
    with open("{}.params".format(prefix), "rb") as fi:
        params = relay.load_param_dict(fi.read())
    return mod, params


def compile_tvm(mod, params, target):
    with tvm.transform.PassContext(opt_level=3):
        if "llvm" not in target:
            desired_layouts = {"nn.conv2d": ["NHWC", "default"]}
            seq = tvm.transform.Sequential(
                [
                    relay.transform.ConvertLayout(desired_layouts),
                    relay.transform.ToMixedPrecision("float16"),
                ]
            )
            mod = seq(mod)

        lib = relay.build(mod, target=target, params=params)
        dev = tvm.device(target, 0)
        return tvm.contrib.graph_executor.GraphModule(lib["default"](dev))


class UNetTVMWrapper(torch.nn.Module):
    def __init__(self, rt_mod, config, in_channels):
        super().__init__()
        self.rt_mod = rt_mod
        self.config = config
        self.in_channels = in_channels
        self.unet_result_type = namedtuple("UNetResult", "sample")

    def forward(self, latent_model_input, timestep, encoder_hidden_states):
        self.rt_mod.set_input("latent_model_input", latent_model_input.numpy())
        self.rt_mod.set_input("timestep", timestep.numpy())
        self.rt_mod.set_input("text_embedding", encoder_hidden_states.numpy())
        self.rt_mod.run()
        return self.unet_result_type(
            torch.from_numpy(self.rt_mod.get_output(0).numpy())
        )


class CLIPTVMWrapper(torch.nn.Module):
    def __init__(self, rt_mod, config, torch_device):
        super().__init__()
        self.rt_mod = rt_mod
        self.config = config
        self.device = torch_device

    def forward(self, input_ids, attention_mask):
        assert attention_mask is None
        self.rt_mod.set_input("text_input_ids", input_ids.numpy())
        self.rt_mod.run()
        return [torch.from_numpy(self.rt_mod.get_output(0).numpy())]


class VAEDecoderTVMWrapper(torch.nn.Module):
    def __init__(self, rt_mod):
        super().__init__()
        self.rt_mod = rt_mod

    def forward(self, latents):
        self.rt_mod.set_input("latents", latents.numpy())
        self.rt_mod.run()
        return torch.from_numpy(self.rt_mod.get_output(0).numpy())


mod_clip, params_clip = deserialize("clip")
mod_unet, params_unet = deserialize("unet")
mod_dec, params_dec = deserialize("dec")

# print(relay.transform.InferType()(mod_unet))

target = "llvm"

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

rt_mod_clip = compile_tvm(mod_clip, params_clip, target)
pipe.text_encoder = CLIPTVMWrapper(
    rt_mod_clip, pipe.text_encoder.config, pipe.text_encoder.device
)

rt_mod_unet = compile_tvm(mod_unet, params_unet, target)
pipe.unet = UNetTVMWrapper(rt_mod_unet, pipe.unet.config, pipe.unet.in_channels)

rt_mod_dec = compile_tvm(mod_dec, params_dec, target)
pipe.vae.decoder = VAEDecoderTVMWrapper(rt_mod_dec)

t1 = time.time()
sample = pipe("Mt. Fuji in the style of Gauguin", num_inference_steps=50)["images"][0]
t2 = time.time()

sample.save("out.png")
print(t2 - t1)
