import tvm
import numpy as np
import torch
import torchvision
import cv2
from tvm.contrib.download import download
from tvm import relay
from tvm.runtime.vm import VirtualMachine


def do_trace(model, in_size=500):
    model_trace = torch.jit.trace(model, torch.rand(1, 3, in_size, in_size))
    model_trace.eval()
    return model_trace


def dict_to_tuple(out_dict):
    if "masks" in out_dict.keys():
        return (out_dict["boxes"], out_dict["scores"], out_dict["labels"], out_dict["masks"])
    return (out_dict["boxes"], out_dict["scores"], out_dict["labels"])


class TraceWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inp):
        out = self.model(inp)
        return dict_to_tuple(out[0])


def get_input(in_size):
    img_path = "test_street_small.jpg"
    img_url = (
        "https://raw.githubusercontent.com/dmlc/web-data/" "master/gluoncv/detection/street_small.jpg"
    )
    download(img_url, img_path)

    img = cv2.imread(img_path).astype("float32")
    img = cv2.resize(img, (in_size, in_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img / 255.0, [2, 0, 1])
    img = np.expand_dims(img, axis=0)
    return img


model_func = torchvision.models.detection.ssd300_vgg16

model = TraceWrapper(model_func(num_classes=50, pretrained_backbone=False))

model.eval()
in_size = 500
img = get_input(in_size)
inp = torch.from_numpy(img)

with torch.no_grad():
    torch_out = model(inp)[0]
    script_module = do_trace(model)
    script_out = script_module(inp)
    mod, params = relay.frontend.from_pytorch(script_module, [("inp", inp.shape)])

target = "llvm"

with tvm.transform.PassContext(opt_level=3):
    vm_exec = relay.vm.compile(mod, target=target, params=params)

dev = tvm.device(target, 0)
vm = VirtualMachine(vm_exec, dev)
vm.set_input("main", **{"inp": img})
vm.run()

tvm_out = vm.get_outputs()[0].numpy()

# Currently TVM outputs are wrong
# Mismatched elements: 773 / 800 (96.6%)
# Max absolute difference: 500.
# Max relative difference: 71.75604
#  x: array([[  0., 500.,   0., 500.],
#        [  0., 500.,   0., 500.],
#        [  0., 500.,   0., 500.],...
#  y: array([[120.42067 , 235.00931 , 199.39113 , 254.69675 ],
#        [163.34209 , 314.05368 , 224.76015 , 494.11188 ],
#        [143.84872 , 258.52884 , 207.84052 , 281.47394 ],...
np.testing.assert_allclose(tvm_out, torch_out, atol=1e-5, rtol=1e-5)
