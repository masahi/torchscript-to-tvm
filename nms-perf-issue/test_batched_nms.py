import torch
import torchvision
import numpy as np
import tvm
from tvm import relay
from tvm.relay import op
from tvm.runtime.vm import VirtualMachine
from tvm.relay.dataflow_pattern import *


def torch_nms(boxes, scores, idxs, max_out_size=1000):
    indices = torchvision.ops.batched_nms(boxes, scores, idxs, 0.7)
    return indices[:max_out_size]


def batched_nms_pattern(boxes, scores, idxs, iou_threshold, num_boxes, indices):
    one = is_constant()
    zero = is_constant()

    # %1824 = cast(%1823, dtype="float32");
    cast = is_op("cast")(idxs)
    mx = is_op("max")(boxes)
    add = is_op("add")(mx, one)
    mul = is_op("multiply")(cast, add)

    dyn_strided_slice = is_op("strided_slice")(mul)
    expand_dims = is_op("expand_dims")(dyn_strided_slice)
    add = is_op("add")(boxes, expand_dims)

    score_expand_dims = is_op("expand_dims")(scores)

    tup = is_tuple([score_expand_dims, add])
    concat = is_op("concatenate")(tup)
    data = is_op("expand_dims")(concat)

    return is_op("vision.non_max_suppression")(
        data, num_boxes, indices, is_constant(), iou_threshold
    )


def convert_batched_nms(boxes, scores, idxs, iou_thres, num_boxes, indices):
    scores = op.expand_dims(scores, axis=-1, num_newaxis=1)
    idxs = op.expand_dims(idxs, axis=-1, num_newaxis=1)
    idxs = op.cast(idxs, "float32")
    data = op.concatenate([idxs, scores, boxes], -1)
    data = op.expand_dims(data, 0, 1)
    top_k = max_out_size = -1
    out = op.vision.non_max_suppression(
        data=data,
        valid_count=num_boxes,
        indices=indices,
        max_output_size=max_out_size,
        iou_threshold=iou_thres,
        force_suppress=False,
        top_k=top_k,
        coord_start=2,
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
        self.num_boxes = wildcard()
        self.indices = wildcard()

        self.pattern = batched_nms_pattern(
            self.boxes,
            self.scores,
            self.idxs,
            self.iou_threshold,
            self.num_boxes,
            self.indices,
        )

    def callback(self, pre, post, node_map):
        boxes = node_map[self.boxes][0]
        scores = node_map[self.scores][0]
        idxs = node_map[self.idxs][0]
        iou_thres = node_map[self.iou_threshold][0]
        num_boxes = node_map[self.num_boxes][0]
        indices = node_map[self.indices][0]
        return convert_batched_nms(boxes, scores, idxs, iou_thres, num_boxes, indices)


def dyn_strided_slice_pattern(inp, end):
    zero = is_constant()
    cast_like = is_op("cast_like")(zero, is_constant())
    less = is_op("less")(is_constant(), cast_like)
    shape_of = is_op("shape_of")(inp)
    cast_like = is_op("cast_like")(shape_of, is_constant())
    add = is_op("add")(is_constant(), cast_like)
    where = is_op("where")(less, add, is_constant())

    return is_op("dyn.strided_slice")(inp, where, end, is_constant())


def topk_after_batch_nms_pattern(
    cond, true_branch, data, valid_count, indices, iou_threshold
):
    batched_nms = is_op("vision.non_max_suppression")(
        data, valid_count, indices, is_constant(), iou_threshold
    )
    indices = is_op("squeeze")(is_tuple_get_item(batched_nms, 0))
    size = is_op("squeeze")(is_tuple_get_item(batched_nms, 1))
    dyn_strided_slice = dyn_strided_slice_pattern(indices, size)
    cast_i64 = is_op("cast")(dyn_strided_slice)

    batched_nms_pattern = is_if(cond, true_branch, cast_i64)

    return is_op("strided_slice")(batched_nms_pattern)


def rewrite_batch_nms_with_max_out_size(
    cond, true_branch, data, valid_count, indices, iou_threshold, post_nms_topk
):
    nms_ret = op.vision.non_max_suppression(
        data=data,
        valid_count=valid_count,
        indices=indices,
        max_output_size=post_nms_topk,
        iou_threshold=iou_threshold,
        force_suppress=False,
        top_k=-1,
        coord_start=2,
        score_index=1,
        id_index=0,
        return_indices=True,
        invalid_to_bottom=False,
    )

    size = op.squeeze(nms_ret[1], axis=[1])
    data_slice = op.squeeze(nms_ret[0], axis=[0])

    ret = op.strided_slice(
        data_slice, begin=relay.const([0]), end=size, slice_mode="size"
    )

    nms_result = op.cast(ret, "int64")

    return relay.If(cond, true_branch, nms_result)


class PostNMSTopKRewrite(DFPatternCallback):
    def __init__(self):
        super().__init__()
        # exprs I want to extract
        self.cond = wildcard()
        self.true_branch = wildcard()
        self.data = wildcard()
        self.valid_count = wildcard()
        self.indices = wildcard()
        self.iou_threshold = wildcard()

        self.pattern = topk_after_batch_nms_pattern(
            self.cond,
            self.true_branch,
            self.data,
            self.valid_count,
            self.indices,
            self.iou_threshold,
        )

    def callback(self, pre, post, node_map):
        print("matched")
        post_nms_topk = post.attrs.end[0].value
        return rewrite_batch_nms_with_max_out_size(
            node_map[self.cond][0],
            node_map[self.true_branch][0],
            node_map[self.data][0],
            node_map[self.valid_count][0],
            node_map[self.indices][0],
            node_map[self.iou_threshold][0],
            post_nms_topk,
        )


boxes_np = np.load("boxes.npy")
scores_np = np.load("scores.npy")
scores_np = scores_np - np.min(scores_np) + 1.0
idxs_np = np.load("idxs.npy")

boxes = torch.from_numpy(boxes_np)
scores = torch.from_numpy(scores_np)
idxs = torch.from_numpy(idxs_np)

trace = torch.jit.trace(torch_nms, [boxes, scores, idxs])

nbox = boxes_np.shape[0]
input_name = "input0"
shape_list = [("boxes", (nbox, 4)), ("scores", (nbox,)), ("idxs", (nbox,))]
mod, params = relay.frontend.from_pytorch(trace, shape_list)

# print(mod["main"])
mod["main"] = rewrite(NMSRewrite(), mod["main"])
mod["main"] = rewrite(PostNMSTopKRewrite(), mod["main"])

# print(mod["main"])

target = "cuda"

with tvm.transform.PassContext(opt_level=3, disabled_pass=["FoldScaleAxis"]):
    vm_exec = relay.vm.compile(mod, target=target, params=params)

######################################################################
# Inference with Relay VM
# -----------------------
ctx = tvm.context(target, 0)
# vm = profiler_vm.VirtualMachineProfiler(vm_exec, ctx)
vm = VirtualMachine(vm_exec, ctx)
vm.set_input("main", **{"boxes": boxes_np, "scores": scores_np, "idxs": idxs_np})
tvm_res = vm.run()
indices_tvm = tvm_res.asnumpy()

boxes = boxes.to("cuda")
scores = scores.to("cuda")
indices_torch = torch_nms(boxes, scores, idxs).cpu().numpy()

print(indices_tvm.shape, indices_torch.shape)
print(np.sum(indices_tvm - indices_torch))

# ftimer = vm.module.time_evaluator("invoke", ctx, number=1, repeat=50)
# print(ftimer("main"))
