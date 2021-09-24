import os
import torch
import numpy as np

import tvm
from tvm import relay, auto_scheduler
import tvm.relay.testing
import tvm.contrib.graph_executor as runtime

###########################################
# Set Tuning Options
# ------------------
# Before tuning, we apply some configurations.

#### DEVICE CONFIG ####
# target = "vulkan -from_device=0"
target = "cuda"
dtype = "float32"

log_file = "bert_large.log"

with open("models/bert_large.json", "r") as fi:
    mod = tvm.ir.load_json(fi.read())
with open("models/bert_large.params", "rb") as fi:
    params = relay.load_param_dict(fi.read())

def auto_schedule(
):
    tasks, task_weights = auto_scheduler.extract_tasks(mod, params, target)

    for idx, task in enumerate(tasks):
        print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
        print(task.compute_dag)

    measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, timeout=100)

    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    # tuner = auto_scheduler.TaskScheduler(tasks, task_weights, load_log_file=log_file)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=50000,  # change this to 20000 to achieve the best performance
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )

    tuner.tune(tune_option)


def evaluate():
    print("Compile...")
    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build_module.build(mod, target=target, params=params)

    # load parameters
    dev = tvm.device(str(target), 0)
    module = runtime.GraphModule(lib["default"](dev))

    batch_size = 8
    inputs = (torch.randint(high=100, size=(batch_size, 128), dtype=torch.int64),
              torch.randint(high=100, size=(batch_size, 128), dtype=torch.int64),
              torch.randint(high=100, size=(batch_size, 128), dtype=torch.int64))

    module.set_input("input_ids", inputs[0].numpy())
    module.set_input("attention_mask", inputs[1].numpy())
    module.set_input("token_type_ids", inputs[2].numpy())
    module.run()
    # print(module.get_output(0))

    # evaluate
    print("Evaluate inference time cost...")
    print(module.benchmark(dev, number=1, repeat=50))

# auto_schedule()
evaluate()
