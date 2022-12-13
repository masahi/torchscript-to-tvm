import time
import pickle
import tempfile
from collections import namedtuple

from diffusers import StableDiffusionPipeline
import torch
from torch.utils.dlpack import to_dlpack, from_dlpack

import tvm
from tvm import relay
from tvm import meta_schedule as ms


def deserialize(prefix):
    with open("{}.json".format(prefix), "r") as fi:
        mod = tvm.ir.load_json(fi.read())

    if prefix == "unet":
        params = {}

        with open("unet.pkl", "rb") as f:
            params_dict = pickle.load(f)
            for k, v in params_dict.items():
                params[k] = tvm.runtime.ndarray.array(v)
    else:
        with open("{}.params".format(prefix), "rb") as fi:
            params = relay.load_param_dict(fi.read())


    return mod, params


def compile_tvm(mod, params, target, tune=False):
    if "llvm" not in target.kind.name:
        desired_layouts = {"nn.conv2d": ["NHWC", "default"]}
        seq = tvm.transform.Sequential(
            [
                relay.transform.ConvertLayout(desired_layouts),
                relay.transform.ToMixedPrecision("float16"),
            ]
        )
        with tvm.transform.PassContext(opt_level=3):
            mod = seq(mod)

    if tune:
        with tempfile.TemporaryDirectory() as work_dir:
            with ms.Profiler() as profiler:
                database = ms.relay_integration.tune_relay(
                    mod=mod,
                    target=target,
                    work_dir=work_dir,
                    max_trials_global=20000,
                    max_trials_per_task=8,
                    num_trials_per_iter=8,
                    strategy="replay-trace",
                    # max_trials_global=20000,
                    # num_trials_per_iter=64,
                    # max_trials_per_task=256,
                    # strategy="evolutionary",
                    params=params,
                )
                lib = ms.relay_integration.compile_relay(
                    database=database,
                    mod=mod,
                    target=target,
                    params=params,
                )
            print(profiler.table())
    else:
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, params=params)

    dev = tvm.device(target.kind.name, 0)
    return tvm.contrib.graph_executor.GraphModule(lib["default"](dev))


def convert_to_ndarray(tensor):
    return tvm.runtime.ndarray.from_dlpack(to_dlpack(tensor))


class UNetTVMWrapper(torch.nn.Module):
    def __init__(self, rt_mod, config, in_channels, torch_device):
        super().__init__()
        self.rt_mod = rt_mod
        self.config = config
        self.in_channels = in_channels
        self.unet_result_type = namedtuple("UNetResult", "sample")
        self.device = torch_device

    def forward(self, latent_model_input, timestep, encoder_hidden_states):
        self.rt_mod.set_input(
            "latent_model_input", convert_to_ndarray(latent_model_input)
        )
        self.rt_mod.set_input("timestep", timestep.numpy())
        self.rt_mod.set_input(
            "text_embedding", convert_to_ndarray(encoder_hidden_states)
        )
        self.rt_mod.run()
        return self.unet_result_type(from_dlpack(self.rt_mod.get_output(0)))


class CLIPTVMWrapper(torch.nn.Module):
    def __init__(self, rt_mod, config, torch_device):
        super().__init__()
        self.rt_mod = rt_mod
        self.config = config
        self.device = torch_device

    def forward(self, input_ids, attention_mask):
        assert attention_mask is None
        self.rt_mod.set_input("text_input_ids", convert_to_ndarray(input_ids))
        self.rt_mod.run()
        return [from_dlpack(self.rt_mod.get_output(0))]


class VAEDecoderTVMWrapper(torch.nn.Module):
    def __init__(self, rt_mod):
        super().__init__()
        self.rt_mod = rt_mod

    def forward(self, latents):
        self.rt_mod.set_input("latents", convert_to_ndarray(latents))
        self.rt_mod.run()
        return from_dlpack(self.rt_mod.get_output(0))


opt_passes = tvm.transform.Sequential([
    relay.transform.SimplifyInference(),
    relay.transform.SimplifyExpr(),
    relay.transform.EliminateCommonSubexpr(),
    relay.transform.CombineParallelDense(min_num_branches=3, to_batch=False)
])

mod_clip, params_clip = deserialize("clip")
mod_unet, params_unet = deserialize("unet")
mod_dec, params_dec = deserialize("dec")

# print(relay.transform.InferType()(mod_unet))

with tvm.transform.PassContext(opt_level=4):
    mod_unet = opt_passes(mod_unet)
    mod_clip = opt_passes(mod_clip)
    mod_dec = opt_passes(mod_dec)

target = tvm.target.Target("llvm")
tune = False

# target = tvm.target.Target("nvidia/geforce-rtx-3070")
# target = tvm.target.Target("rocm")
# tune = True

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

if "llvm" not in target.kind.name:
    pipe.to("cuda")

rt_mod_clip = compile_tvm(mod_clip, params_clip, target, tune)
rt_mod_unet = compile_tvm(mod_unet, params_unet, target, tune)
rt_mod_dec = compile_tvm(mod_dec, params_dec, target, tune)

pipe.text_encoder = CLIPTVMWrapper(
    rt_mod_clip, pipe.text_encoder.config, pipe.text_encoder.device
)
pipe.unet = UNetTVMWrapper(
    rt_mod_unet, pipe.unet.config, pipe.unet.in_channels, pipe.unet.device
)
pipe.vae.decoder = VAEDecoderTVMWrapper(rt_mod_dec)

t1 = time.time()
sample = pipe("Mt. Fuji in the style of Gauguin", num_inference_steps=50)["images"][0]
t2 = time.time()

sample.save("out.png")
print(t2 - t1)
