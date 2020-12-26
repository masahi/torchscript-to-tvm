import tvm
from tvm import relay, auto_scheduler
from tvm.relay.frontend.pytorch_utils import NMSRewrite
from tvm.contrib.download import download
from tvm.runtime.vm import VirtualMachine
from tvm.relay.dataflow_pattern import *

import numpy as np
import cv2


import torch
import torchvision

in_size = 300

input_shape = (1, 3, in_size, in_size)


def do_trace(model, inp):
    model_trace = torch.jit.trace(model, inp)
    model_trace.eval()
    return model_trace


def dict_to_tuple(out_dict):
    if "masks" in out_dict.keys():
        return (
            out_dict["boxes"],
            out_dict["scores"],
            out_dict["labels"],
            out_dict["masks"],
        )
    return out_dict["boxes"], out_dict["scores"], out_dict["labels"]


class TraceWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inp):
        out = self.model(inp)
        return dict_to_tuple(out[0])


def get_input():
    img_path = "test_street_small.jpg"
    img_url = (
        "https://raw.githubusercontent.com/dmlc/web-data/"
        "master/gluoncv/detection/street_small.jpg"
    )
    download(img_url, img_path)

    img = cv2.imread(img_path).astype("float32")
    img = cv2.resize(img, (in_size, in_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img / 255.0, [2, 0, 1])
    img = np.expand_dims(img, axis=0)
    return img




input_name = "input0"
img = get_input()

# model_func = torchvision.models.detection.maskrcnn_resnet50_fpn
# model = TraceWrapper(model_func(pretrained=True))
# model.eval()

# inp = torch.from_numpy(img)

# with torch.no_grad():
#     out = model(inp)
#     script_module = do_trace(model, inp)


# shape_list = [(input_name, input_shape)]
# mod, params = relay.frontend.from_pytorch(script_module, shape_list)

# with open("maskrcnn_mod.json", "w") as fo:
#     fo.write(tvm.ir.save_json(mod))
# with open("maskrcnn.params", "wb") as fo:
#     fo.write(relay.save_param_dict(params))

with open("maskrcnn_mod.json", "r") as fi:
    mod = tvm.ir.load_json(fi.read())
with open("maskrcnn.params", "rb") as fi:
    params = relay.load_param_dict(fi.read())

mod["main"] = rewrite(NMSRewrite(), mod["main"])
print(mod["main"])

# with tvm.transform.PassContext(opt_level=3, disabled_pass=["FoldScaleAxis"]):
#     vm_exec = relay.vm.compile(mod, target=target, params=params)

# ######################################################################
# # Inference with Relay VM
# # -----------------------
# ctx = tvm.cpu()
# vm = VirtualMachine(vm_exec, ctx)
# vm.set_input("main", **{input_name: img})
# tvm_res = vm.run()

# ######################################################################
# # Get boxes with score larger than 0.9
# # ------------------------------------
# score_threshold = 0.9
# boxes = tvm_res[0].asnumpy().tolist()
# valid_boxes = []
# for i, score in enumerate(tvm_res[1].asnumpy().tolist()):
#     if score > score_threshold:
#         valid_boxes.append(boxes[i])
#     else:
#         break

# print("Get {} valid boxes".format(len(valid_boxes)))
