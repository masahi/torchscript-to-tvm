import torch
import torchvision
import numpy as np
import tvm
from tvm import relay
from tvm.runtime.vm import VirtualMachine


iou_threshold = np.asscalar(np.load("iou_threshold.npy")[0])


def torch_nms(boxes, scores):
    return torchvision.ops.nms(boxes, scores, iou_threshold)


boxes_np = np.load("boxes.npy")
scores_np = np.load("scores.npy")
scores_np = scores_np - np.min(scores_np) + 1.0
# scores_np = np.clip(scores_np, 0, np.max(scores_np)+1)
# scores_np = np.random.rand(boxes_np.shape[0]).astype(scores_np.dtype)

boxes = torch.from_numpy(boxes_np)
scores = torch.from_numpy(scores_np)

trace = torch.jit.trace(torch_nms, [boxes, scores])

input_name = "input0"
shape_list = [("boxes", boxes_np.shape), ("scores", scores_np.shape)]
mod, params = relay.frontend.from_pytorch(trace, shape_list)

target = "cuda"

with tvm.transform.PassContext(opt_level=3, disabled_pass=["FoldScaleAxis"]):
    vm_exec = relay.vm.compile(mod, target=target, params=params)

######################################################################
# Inference with Relay VM
# -----------------------
ctx = tvm.context(target, 0)
# vm = profiler_vm.VirtualMachineProfiler(vm_exec, ctx)
vm = VirtualMachine(vm_exec, ctx)
vm.set_input("main", **{"boxes": boxes_np, "scores": scores_np})
tvm_res = vm.run()
indices_tvm = tvm_res.asnumpy()

boxes = boxes.to("cuda")
scores = scores.to("cuda")
indices_torch = torch_nms(boxes, scores).cpu().numpy()

print(indices_tvm.shape, indices_torch.shape)
print(np.sum(indices_tvm - indices_torch))
