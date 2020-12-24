import tvm
from tvm import relay, auto_scheduler
from tvm.runtime import profiler_vm
from tvm.runtime.vm import VirtualMachine
from tvm.contrib.download import download

import numpy as np
import cv2

# PyTorch imports
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


def benchmark_torch(model, inp, num_iters):
    model.to("cuda")
    inp = inp.to("cuda")

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


num_iters = 50

model_func = torchvision.models.detection.maskrcnn_resnet50_fpn
model = TraceWrapper(model_func(pretrained=True, rpn_pre_nms_top_n_test=200))

model.eval()
img = get_input(in_size)
inp = torch.from_numpy(img)

with torch.no_grad():
    out = model(inp)
    script_module = do_trace(model, inp)


def auto_schedule():
    input_name = "input0"
    shape_list = [(input_name, input_shape)]
    mod, params = relay.frontend.from_pytorch(script_module, shape_list)

    target = "cuda"

    tasks, task_weights = auto_scheduler.extract_tasks(mod, params, target)

    for idx, task in enumerate(tasks):
        print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
        print(task.compute_dag)

    log_file = "maskrcnn.log"
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, timeout=10)

    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=20000,  # change this to 20000 to achieve the best performance
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )

    tuner.tune(tune_option)


def bench_tvm():
    ######################################################################
    # Import the graph to Relay
    # -------------------------
    input_name = "input0"
    shape_list = [(input_name, input_shape)]
    mod, params = relay.frontend.from_pytorch(script_module, shape_list)

    target = "cuda"

    # log_file = "../auto_scheduler/resnet-50-NHWC-B1.json"
    # with auto_scheduler.ApplyHistoryBest(log_file):
    with tvm.transform.PassContext(opt_level=3, disabled_pass=["FoldScaleAxis"]):
        vm_exec = relay.vm.compile(mod, target=target, params=params)

    ######################################################################
    # Inference with Relay VM
    # -----------------------
    ctx = tvm.context(target, 0)
    # vm = profiler_vm.VirtualMachineProfiler(vm_exec, ctx)
    vm = VirtualMachine(vm_exec, ctx)
    vm.set_input("main", **{input_name: img})
    vm.run()
    # print("\n{}".format(vm.get_stat()))

    ftimer = vm.module.time_evaluator("invoke", ctx, number=1, repeat=num_iters)
    print(ftimer("main"))


# benchmark_torch(model, inp, num_iters)
# bench_tvm()
auto_schedule()
