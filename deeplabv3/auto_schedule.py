import tvm
from tvm import relay, auto_scheduler
from tvm.runtime import profiler_vm
from tvm.runtime.vm import VirtualMachine

import numpy as np

# PyTorch imports
import torch
import torchvision
from torch import nn


class TraceWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inp):
        out = self.model(inp)
        return out["out"]


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
deeplabv3 = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True)
model = TraceWrapper(deeplabv3.eval())
model.eval()
inp = torch.rand(8, 3, 512, 512)

with torch.no_grad():
    trace = torch.jit.trace(model, inp)
    torch_res = model(inp)

target = "cuda"
log_file = "deeplabvv3_b8.log"

def auto_schedule():
    mod, params = relay.frontend.from_pytorch(trace, [('input', inp.shape)])

    with tvm.transform.PassContext(opt_level=3):
        desired_layouts = {'nn.conv2d': ['NHWC', 'default']}
        seq = tvm.transform.Sequential([relay.transform.ConvertLayout(desired_layouts)])
        mod = seq(mod)

    with open("deeplabv3.txt", "w") as f:
        f.write(str(relay.transform.InferType()(mod)))

    tasks, task_weights = auto_scheduler.extract_tasks(mod, params, target)

    for idx, task in enumerate(tasks):
        print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
        print(task.compute_dag)

    measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, timeout=100)
    # tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tuner = auto_scheduler.TaskScheduler(tasks, task_weights, load_log_file=log_file)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=50000,  # change this to 20000 to achieve the best performance
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )

    tuner.tune(tune_option)


def bench_tvm():
    mod, params = relay.frontend.from_pytorch(trace, [('input', inp.shape)])

    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
            desired_layouts = {'nn.conv2d': ['NHWC', 'default']}
            seq = tvm.transform.Sequential([relay.transform.ConvertLayout(desired_layouts)])
            mod = seq(mod)
            json, lib, params = relay.build(mod, target=target, params=params)

    ctx = tvm.device(target, 0)
    runtime = tvm.contrib.graph_executor.create(json, lib, ctx)
    runtime.set_input(**params)
    runtime.set_input("input", inp.numpy())
    runtime.run()

    ftimer = runtime.module.time_evaluator("run", ctx, number=1, repeat=50)
    prof_res = np.array(ftimer().results) * 1000
    print(prof_res)
    print(np.mean(prof_res))


# benchmark_torch(model, inp, num_iters)
bench_tvm()
# auto_schedule()
