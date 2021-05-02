import tvm
from tvm import relay, auto_scheduler
from tvm.runtime import profiler_vm
from tvm.runtime.vm import VirtualMachine

import sys
sys.path.append("../../../deep/detr/")

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


def get_torch_outputs(model, inp):
    with torch.no_grad():
        raw_outputs = model(inp)
        outputs, _ = torch.jit._flatten(raw_outputs)
        return [output.cpu().numpy() for output in outputs]


num_iters = 50

model = TraceWrapper(detr_resnet50(pretrained=True).eval())
model.eval()
inp = torch.rand(1, 3, 750, 800)

# model = torchvision.models.resnet50(pretrained=True)
# model.eval()
# inp = torch.rand(1, 3, 224, 224)

with torch.no_grad():
    trace = torch.jit.trace(model, inp)
    torch_res = model(inp)

mod, params = relay.frontend.from_pytorch(trace, [('input', inp.shape)])


def auto_schedule():
    target = "opencl"

    # mod, params = relay.frontend.from_pytorch(trace, [('input', inp.shape)])

    # with open("detr_mod.json", "w") as fo:
    #     fo.write(tvm.ir.save_json(mod))
    # with open("detr.params", "wb") as fo:
    #     fo.write(relay.save_param_dict(params))

    with open("detr_mod.json", "r") as fi:
        mod = tvm.ir.load_json(fi.read())
    with open("detr.params", "rb") as fi:
        params = relay.load_param_dict(fi.read())

    with tvm.transform.PassContext(opt_level=3):
        desired_layouts = {'nn.conv2d': ['NHWC', 'default']}
        seq = tvm.transform.Sequential([relay.transform.ConvertLayout(desired_layouts)])
        mod = seq(mod)

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
    target = "opencl"

    mod, params = relay.frontend.from_pytorch(trace, [('input', inp.shape)])

    # with auto_scheduler.ApplyHistoryBest("logs/detr_gen11_nchw.log"):
    #     with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
    #         json, lib, params = relay.build(mod, target=target, params=params)

    with auto_scheduler.ApplyHistoryBest("logs/detr_gen11_nhwc.log"):
        with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
            desired_layouts = {'nn.conv2d': ['NHWC', 'default']}
            seq = tvm.transform.Sequential([relay.transform.ConvertLayout(desired_layouts)])
            mod = seq(mod)
            json, lib, params = relay.build(mod, target=target, params=params)

    ctx = tvm.device(target, 0)
    runtime = tvm.contrib.graph_executor.create(json, lib, ctx)
    runtime.set_input(**params)
    runtime.set_input("input", inp.numpy())
    import time
    t1 = time.time()
    runtime.run()
    print("run finished in", time.time() - t1)

    tvm_results = [runtime.get_output(i).asnumpy() for i in [0, 1]]
    pt_results = get_torch_outputs(model, inp)

    for pt_res, tvm_res in zip(pt_results, tvm_results):
        print(np.mean(np.abs(pt_res - tvm_res)))

    ftimer = runtime.module.time_evaluator("run", ctx, number=1, repeat=20)
    prof_res = np.array(ftimer().results) * 1000
    print(prof_res)
    print(np.mean(prof_res))


def bench_tvm_vm():
    target = "opencl"

    mod, params = relay.frontend.from_pytorch(trace, [('input', inp.shape)])

    # with auto_scheduler.ApplyHistoryBest("logs/detr_gen11_nchw.log"):
    #     with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
    #         json, lib, params = relay.build(mod, target=target, params=params)

    with auto_scheduler.ApplyHistoryBest("logs/detr_gen11_nhwc.log"):
        with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
            desired_layouts = {'nn.conv2d': ['NHWC', 'default']}
            seq = tvm.transform.Sequential([relay.transform.ConvertLayout(desired_layouts)])
            mod = seq(mod)
            vm_exec = relay.vm.compile(mod, target=target, params=params)
            print("compile finished")

    ctx = tvm.device(target, 0)

    vm = VirtualMachine(vm_exec, ctx)
    vm.set_input("main", **{"input": inp.numpy()})
    vm.run()

    ftimer = vm.module.time_evaluator("invoke", ctx, number=1, repeat=20)
    res = ftimer("main")
    print(res)


# benchmark_torch(model, inp, num_iters)
bench_tvm()
# auto_schedule()
