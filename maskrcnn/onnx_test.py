import tvm
import onnx
from tvm import relay, auto_scheduler
from tvm.runtime.vm import VirtualMachine
from tvm.contrib.download import download

import numpy as np
import cv2

# PyTorch imports
import torch
import torchvision

in_size = 300

input_shape = (1, 3, in_size, in_size)


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


model_func = torchvision.models.detection.maskrcnn_resnet50_fpn
# model_func = torchvision.models.detection.fasterrcnn_resnet50_fpn
model = model_func(pretrained=True)

img = get_input(in_size)
inp = torch.from_numpy(img)


def test_onnx():
    torch.onnx.export(model, inp, "maskrcnn.onnx", opset_version=11)
    onnx_model = onnx.load("maskrcnn.onnx")
    input_name = "inputs"
    shape_dict = {input_name: inp.shape}
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)


test_onnx()
