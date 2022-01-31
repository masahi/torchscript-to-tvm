# PyTorch imports
import torch
import torchvision

import tvm
from tvm import relay, auto_scheduler
from tvm.runtime import profiler_vm
from tvm.runtime.vm import VirtualMachine

import numpy as np

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
detr = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=False)
model = TraceWrapper(detr.eval())
model.eval()
inp = torch.rand(1, 3, 750, 800)

with torch.no_grad():
    trace = torch.jit.trace(model, inp)
    torch_res = model(inp)

use_fp16 = True
target = "vulkan -from_device=0"
log_file = "logs/radv_aco_fp16_wave32_6600xt.log"
# target = "rocm"
# log_file = "logs/rocm_6600xt_fp16.log"

def auto_schedule():
    # mod, params = relay.frontend.from_pytorch(trace, [('input', inp.shape)])

    # with open("detr_mod.json", "w") as fo:
    #     fo.write(tvm.ir.save_json(mod))
    # with open("detr.params", "wb") as fo:
    #     fo.write(relay.save_param_dict(params))

    with open("detr_mod.json", "r") as fi:
        mod = tvm.ir.load_json(fi.read())
    with open("detr.params", "rb") as fi:
        params = relay.load_param_dict(fi.read())

    if use_fp16:
        from tvm.relay.transform import InferType, ToMixedPrecision, mixed_precision
        mod = ToMixedPrecision("float16")(mod)

    with tvm.transform.PassContext(opt_level=3):
        desired_layouts = {'nn.conv2d': ['NHWC', 'default']}
        seq = tvm.transform.Sequential([relay.transform.ConvertLayout(desired_layouts)])
        mod = seq(mod)

    tasks, task_weights = auto_scheduler.extract_tasks(mod, params, target)

    for idx, task in enumerate(tasks):
        print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
        print(task.compute_dag)

    measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, timeout=100)
    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    # tuner = auto_scheduler.TaskScheduler(tasks, task_weights, load_log_file=log_file)x
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=50000,  # change this to 20000 to achieve the best performance
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )

    tuner.tune(tune_option)


def bench_tvm():
    mod, params = relay.frontend.from_pytorch(trace, [('input', inp.shape)])

    if use_fp16:
        from tvm.relay.transform import InferType, ToMixedPrecision, mixed_precision
        mod = ToMixedPrecision("float16")(mod)
        # print(mod)

    # with tvm.transform.PassContext(opt_level=3):
    #     desired_layouts = {'nn.conv2d': ['NHWC', 'default']}
    #     seq = tvm.transform.Sequential([relay.transform.ConvertLayout(desired_layouts)])
    #     mod = seq(mod)
    #     json, lib, params = relay.build(mod, target=target, params=params)

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

    # with open("rocm_fp16_asm.s", "w") as f:
    #     f.write(str(lib.imported_modules[0].get_source("asm")))

    tvm_results = [runtime.get_output(i).asnumpy() for i in [0, 1]]
    pt_results = get_torch_outputs(model, inp)

    for pt_res, tvm_res in zip(pt_results, tvm_results):
        print(np.mean(np.abs(pt_res - tvm_res)))

    ftimer = runtime.module.time_evaluator("run", ctx, number=1, repeat=50)
    prof_res = np.array(ftimer().results) * 1000
    print(prof_res)
    print(np.mean(prof_res))


# benchmark_torch(model, inp, num_iters)
bench_tvm()
# auto_schedule()
