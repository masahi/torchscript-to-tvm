import tvm
from tvm import relay
from torch.quantization import get_default_qconfig
from torch.quantization.quantize_fx import prepare_fx, convert_fx
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


model = yolov5l(export_friendly=True, pretrained=True)
model.eval()
inp = torch.Tensor(np.random.uniform(0.0, 250.0, size=(1, 3, in_size, in_size)))

qconfig = get_default_qconfig("fbgemm")
qconfig_dict = {"": qconfig}

prepared_model = model

qmodel = model
qmodel.model.backbone =  convert_fx(prepare_fx(model.model.backbone, qconfig_dict))
# qmodel.model.head.head =  convert_fx(prepare_fx(model.model.head.head, qconfig_dict))

qmodel(inp)

model = TraceWrapper(qmodel)

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
