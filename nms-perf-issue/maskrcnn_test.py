import tvm
from tvm import relay
from tvm.contrib.download import download
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
        return out_dict["boxes"], out_dict["scores"], out_dict["labels"], out_dict["masks"]
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
        "https://raw.githubusercontent.com/dmlc/web-data/" "master/gluoncv/detection/street_small.jpg"
    )
    download(img_url, img_path)

    img = cv2.imread(img_path).astype("float32")
    img = cv2.resize(img, (in_size, in_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img / 255.0, [2, 0, 1])
    img = np.expand_dims(img, axis=0)
    return img


def batched_nms_pattern():
    # exprs I want to extract
    boxes = wildcard()
    scores = wildcard()
    idxs = wildcard()

    one = is_constant()
    zero = is_constant()

    # %1796 = expand_dims(%1795, axis=-1);
    score_expand_dims = is_op("expand_dims")(scores)

    # %1824 = cast(%1823, dtype="float32");
    cast = is_op("cast")(idxs)
    mx = is_op("max")(boxes)
    add = is_op("add")(mx, one)
    mul = is_op("multiply")(cast, add)

    # %1828 = cast_like(0, meta[relay.Constant][127]);
    cast_like = is_op("cast_like")(zero, is_constant())
    less = is_op("less")(is_constant(), cast_like)
    shape_of = is_op("shape_of")(mul)
    cast_like = is_op("cast_like")(shape_of, is_constant())
    add = is_op("add")(is_constant(), cast_like)
    where = is_op("where")(less, add, is_constant())
    shape_of = is_op("shape_of")(mul)
    cast = is_op("cast")(shape_of)

    # %1836 = dyn.strided_slice(%1827, %1833, %1835, meta[relay.Constant][128], begin=None, end=None, strides=None);
    dyn_strided_slice = is_op("dyn.strided_slice")(mul, where, cast, is_constant())

    expand_dims = is_op("expand_dims")(dyn_strided_slice)
    add = is_op("add")(boxes, expand_dims)
    tup = is_tuple([score_expand_dims, add])
    concat = is_op("concatenate")(tup)
    expand_dims = is_op("expand_dims")(concat)

    # %1842 = vision.get_valid_counts(%1841, -1f, meta[relay.attrs.GetValidCountsAttrs][1]);
    return is_op("vision.get_valid_counts")(expand_dims, is_constant(), wildcard())


model_func = torchvision.models.detection.maskrcnn_resnet50_fpn
model = TraceWrapper(model_func(pretrained=True))
model.eval()

img = get_input()
inp = torch.from_numpy(img)

with torch.no_grad():
    out = model(inp)
    script_module = do_trace(model, inp)

input_name = "input0"
shape_list = [(input_name, input_shape)]
mod, params = relay.frontend.from_pytorch(script_module, shape_list)
# print(mod["main"])

pat = batched_nms_pattern()
print(pat.match(mod["main"].body))
