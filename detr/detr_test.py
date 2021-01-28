import tvm
from tvm import relay, auto_scheduler
from tvm.runtime import profiler_vm
from tvm.runtime.vm import VirtualMachine

import sys
sys.path.append("../../../ml/detr/")

from hubconf import detr_resnet50

import numpy as np
import cv2

# PyTorch imports
import torch
import torchvision

in_size = 300

input_shape = (1, 3, in_size, in_size)


class TraceWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inp):
        out = self.model(inp)
        return (out['pred_logits'], out['pred_boxes'])


def benchmark_torch(model, inp, num_iters):
    model.to("cuda")
    inp = inp.to("cuda")

    # torch.backends.cudnn.benchmark = True

    with torch.no_grad():
        for i in range(3):
            model(inp)
        torch.cuda.synchronize()

        import time
        t1 = time.time()
        for i in range(num_iters):
            model(inp)
        torch.cuda.synchronize()
        t2 = time.time()

        print("torch elapsed", (t2 - t1) / num_iters)


num_iters = 50

model = TraceWrapper(detr_resnet50(pretrained=False).eval())

model.eval()
inp = torch.rand(1, 3, 750, 800)

# with torch.no_grad():
#     trace = torch.jit.trace(model, inp)

# mod, params = relay.frontend.from_pytorch(trace, [('input', inp.shape)])

# with open("detr_mod.json", "w") as fo:
#     fo.write(tvm.ir.save_json(mod))
# with open("detr.params", "wb") as fo:
#     fo.write(relay.save_param_dict(params))

with open("detr_mod.json", "r") as fi:
    mod = tvm.ir.load_json(fi.read())
with open("detr.params", "rb") as fi:
    params = relay.load_param_dict(fi.read())


def auto_schedule():
    target = "rocm"

    tasks, task_weights = auto_scheduler.extract_tasks(mod, params, target)

    for idx, task in enumerate(tasks):
        print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
        print(task.compute_dag)

    log_file = "detr.log"
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, timeout=100)

    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    # tuner = auto_scheduler.TaskScheduler(tasks, task_weights, load_log_file=log_file)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=50000,  # change this to 20000 to achieve the best performance
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )

    tuner.tune(tune_option)


def bench_tvm():
    target = "rocm"

    with auto_scheduler.ApplyHistoryBest("logs/detr_rocm.log"):
        with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
            vm_exec = relay.vm.compile(mod, target=target, params=params)

    # with tvm.transform.PassContext(opt_level=3):
    #    vm_exec = relay.vm.compile(mod, target=target, params=params)

    ######################################################################
    # Inference with Relay VM
    # -----------------------
    ctx = tvm.context(target, 0)
    # vm = profiler_vm.VirtualMachineProfiler(vm_exec, ctx)
    vm = VirtualMachine(vm_exec, ctx)
    vm.set_input("main", **{"input": inp.numpy()})
    vm.run()
    # print("\n{}".format(vm.get_stat()))

    ftimer = vm.module.time_evaluator("invoke", ctx, number=1, repeat=num_iters)
    print(ftimer("main"))


# benchmark_torch(model, inp, num_iters)
bench_tvm()
# auto_schedule()
