import time
from collections import namedtuple

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
    class UNetWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, latent_model_input, timestep, encoder_hidden_states):
            return self.model(
                latent_model_input, timestep, encoder_hidden_states
            ).sample

    class CLIPWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids):
            return self.model(input_ids)[0]

    with torch.no_grad():
        clip_traced = torch.jit.trace(
            CLIPWrapper(pipe.text_encoder), torch.ones(1, 77, dtype=torch.int64)
        )
        unet_traced = torch.jit.trace(
            UNetWrapper(pipe.unet),
            [torch.randn(2, 4, 64, 64), torch.tensor(1), torch.randn(2, 77, 768)],
        )
        vae_dec_traced = torch.jit.trace(pipe.vae.decoder, torch.randn(1, 4, 64, 64))

    mod_clip, params_clip = relay.frontend.from_pytorch(
        clip_traced, [("text_input_ids", (1, 77))]
    )
    mod_unet, params_unet = relay.frontend.from_pytorch(
        unet_traced,
        [
            ("latent_model_input", (2, 4, 64, 64)),
            ("timestep", ()),
            ("text_embedding", (2, 77, 768)),
        ],
    )
    mod_vae_dec, params_vae_dec = relay.frontend.from_pytorch(
        vae_dec_traced, [("latents", (1, 4, 64, 64))]
    )

    serialize(mod_clip, params_clip, "clip")
    serialize(mod_unet, params_unet, "unet")
    serialize(mod_vae_dec, params_vae_dec, "dec")


def compile_tvm(mod, params, target):
    with tvm.transform.PassContext(opt_level=3):
        # desired_layouts = {'nn.conv2d': ['NHWC', 'default']}
        # seq = tvm.transform.Sequential([relay.transform.ConvertLayout(desired_layouts)])
        # mod = seq(mod)
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
        self.dec_result_type = namedtuple("DecResult", "sample")
        # self.config = config
        # self.device = torch_device

    def forward(self, latents):
        self.rt_mod.set_input("latents", latents.numpy())
        self.rt_mod.run()
        return self.dec_result_type(torch.from_numpy(self.rt_mod.get_output(0).numpy()))


# export_models()

# mod_clip, params_clip = deserialize("clip")
# mod_unet, params_unet = deserialize("unet")
mod_dec, params_dec = deserialize("dec")

# print(relay.transform.InferType()(mod_unet))

target = "llvm"

# rt_mod_clip = compile_tvm(mod_clip, params_clip, target)
# pipe.text_encoder = CLIPTVMWrapper(rt_mod_clip, pipe.text_encoder.config, pipe.text_encoder.device)

# rt_mod_unet = compile_tvm(mod_unet, params_unet, target)
# pipe.unet = UNetTVMWrapper(rt_mod_unet, pipe.unet.config, pipe.unet.in_channels)

rt_mod_dec = compile_tvm(mod_dec, params_dec, target)

import time

t1 = time.time()
sample = pipe("Mt. Fuji in the style of Gauguin", num_inference_steps=50)["images"][0]
t2 = time.time()

sample.save("out.png")
print(t2 - t1)
