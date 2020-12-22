import tvm
from tvm import relay
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


def batched_nms_pattern(boxes, scores, idxs, iou_threshold):
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
    get_valid_counts_out = is_op("vision.get_valid_counts")(expand_dims, is_constant())
    data = is_tuple_get_item(get_valid_counts_out, 1)
    valid_counts = is_tuple_get_item(get_valid_counts_out, 0)
    indices = is_tuple_get_item(get_valid_counts_out, 2)
    return is_op("vision.non_max_suppression")(
        data, valid_counts, indices, is_constant(), iou_threshold
    )


def convert_batched_nms(boxes, scores, idxs, iou_thres):
    scores = relay.expand_dims(scores, axis=-1, num_newaxis=1)
    idxs = relay.expand_dims(idxs, axis=-1, num_newaxis=1)
    idxs = relay.cast(idxs, "float32")
    data = relay.concatenate([idxs, scores, boxes], -1)
    data = relay.expand_dims(data, 0, 1)
    ct, data, indices = relay.op.vision.get_valid_counts(
        data, score_threshold=-1.0, id_index=0, score_index=1
    )
    top_k = max_out_size = -1
    out = relay.op.vision.non_max_suppression(
        data=data,
        valid_count=ct,
        indices=indices,
        max_output_size=max_out_size,
        iou_threshold=iou_thres,
        force_suppress=True,
        top_k=top_k,
        coord_start=1,
        score_index=1,
        id_index=0,
        return_indices=True,
        invalid_to_bottom=False,
    )
    return out.tuple_value


class NMSRewrite(DFPatternCallback):
    def __init__(self):
        super().__init__()
        # exprs I want to extract
        self.boxes = wildcard()
        self.scores = wildcard()
        self.idxs = wildcard()
        self.iou_threshold = wildcard()

        self.pattern = batched_nms_pattern(
            self.boxes, self.scores, self.idxs, self.iou_threshold
        )

    def callback(self, pre, post, node_map):
        print("matched")
        boxes = node_map[self.boxes][0]
        scores = node_map[self.scores][0]
        idxs = node_map[self.idxs][0]
        iou_thres = node_map[self.iou_threshold][0]
        return convert_batched_nms(boxes, scores, idxs, iou_thres)


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
# print(mod["main"])
with open("maskrcnn_mod.json", "r") as fi:
    mod = tvm.ir.load_json(fi.read())
with open("maskrcnn.params", "rb") as fi:
    params = relay.load_param_dict(fi.read())

mod["main"] = rewrite(NMSRewrite(), mod["main"])
# print(mod["main"])

target = "llvm"

with tvm.transform.PassContext(opt_level=3, disabled_pass=["FoldScaleAxis"]):
    vm_exec = relay.vm.compile(mod, target=target, params=params)

######################################################################
# Inference with Relay VM
# -----------------------
ctx = tvm.cpu()
vm = VirtualMachine(vm_exec, ctx)
vm.set_input("main", **{input_name: img})
tvm_res = vm.run()

######################################################################
# Get boxes with score larger than 0.9
# ------------------------------------
score_threshold = 0.9
boxes = tvm_res[0].asnumpy().tolist()
valid_boxes = []
for i, score in enumerate(tvm_res[1].asnumpy().tolist()):
    if score > score_threshold:
        valid_boxes.append(boxes[i])
    else:
        break

print("Get {} valid boxes".format(len(valid_boxes)))
