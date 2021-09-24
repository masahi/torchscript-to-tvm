import tvm
from tvm import relay
from tvm.runtime.vm import VirtualMachine

import numpy as np
import cv2

import torch
from torch import nn


in_size = 416

input_shape = (1, 3, in_size, in_size)


def do_trace(model, inp):
    model_trace = torch.jit.trace(model, inp)
    model_trace.eval()
    return model_trace


def dict_to_tuple(out_dict):
    if "masks" in out_dict.keys():
        return out_dict["boxes"], out_dict["scores"], out_dict["labels"], out_dict["masks"]
    return out_dict["boxes"], out_dict["scores"], out_dict["labels"]


class TraceWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inp):
        out = self.model(inp)
        return dict_to_tuple(out[0])


from yolort.models import yolov5l


model_func = yolov5l(export_friendly=True, pretrained=True)


model = TraceWrapper(model_func)

model.eval()
inp = torch.Tensor(np.random.uniform(0.0, 250.0, size=(1, 3, in_size, in_size)))

with torch.no_grad():
    out = model(inp)
    script_module = do_trace(model, inp)


img = cv2.imread("bus.jpg")

img = img.astype("float32")
img = cv2.resize(img, (in_size, in_size))

img = np.transpose(img / 255.0, [2, 0, 1])
img = np.expand_dims(img, axis=0)


input_name = "input0"
shape_list = [(input_name, input_shape)]
mod, params = relay.frontend.from_pytorch(script_module, shape_list)

# from tvm.relay.transform import InferType, ToMixedPrecision, mixed_precision
# mod = ToMixedPrecision("float16")(mod)

# print(relay.transform.InferType()(mod))

target = "cuda"
# target = "llvm"

with tvm.transform.PassContext(opt_level=3):
    vm_exec = relay.vm.compile(mod, target=target, params=params)

ctx = tvm.device(target, 0)
vm = VirtualMachine(vm_exec, ctx)
vm.set_input("main", **{input_name: img})
tvm_res = vm.run()

with torch.no_grad():
    torch_res = model(torch.from_numpy(img))

for i in range(3):
    print(np.max(np.abs(torch_res[i].numpy() - tvm_res[i].asnumpy())))
