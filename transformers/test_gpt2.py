import torch
from transformers import GPT2Model, GPT2LMHeadModel
from transformers.modeling_utils import Conv1D

import numpy as np
import os


def _conv1d_to_linear(module):
    in_size, out_size = module.weight.shape
    linear = torch.nn.Linear(in_size, out_size)
    linear.weight.data = module.weight.data.T.contiguous()
    linear.bias.data = module.bias.data
    return linear


def conv1d_to_linear(model):
    for name in list(model._modules):
        module = model._modules[name]
        if isinstance(module, Conv1D):
            linear = _conv1d_to_linear(module)
            model._modules[name] = linear
        else:
            conv1d_to_linear(module)


MODELS = [
    (GPT2Model, "gpt2"),
    # (GPT2LMHeadModel, 'gpt2'),
]


def gpt2_test():
    for model_class, pretrained_weights in MODELS:
        # Load pretrained model/tokenizer
        model = model_class.from_pretrained(pretrained_weights)
        model.eval()
        input_ids_1 = torch.ones(1, 1, 128, dtype=torch.int64)

        conv1d_to_linear(model)

        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )

        # pt_model = quantized_model
        pt_model = model

        with torch.no_grad():
            pt_outputs = pt_model(input_ids_1)

        torch_outputs = []

        for out in pt_outputs:
            if isinstance(out, tuple):
                for o in out:
                    if isinstance(o, torch.Tensor):
                        print(o.shape)
                        torch_outputs.append(o.numpy())
            if isinstance(out, torch.Tensor):
                print(out.shape)
                torch_outputs.append(out.numpy())

        script_module = torch.jit.trace(pt_model, input_ids_1).eval()

        import tvm
        from tvm import relay
        from tvm.runtime.vm import VirtualMachine

        input_name = "input_ids"
        input_shapes = [(input_name, (input_ids_1.shape, "int64"))]
        mod, params = relay.frontend.from_pytorch(script_module, input_shapes)

        target = "llvm"

        print("inferred type")
        print(relay.transform.InferType()(mod))

        with tvm.transform.PassContext(opt_level=3):
            # opt_mod, opt_params = relay.optimize(mod, target="llvm -mcpu=cascadelake -libs=mkl", params=params)
            # print(opt_mod["main"])
            # vm_exec = relay.vm.compile(mod, target=target, params=params)
            lib = relay.build(mod, target=target, params=params)

        inp = input_ids_1.numpy()

        # vm = VirtualMachine(vm_exec, ctx)
        # vm.set_input("main", **{input_name: inp})
        # tvm_res = vm.run()

        runtime = tvm.contrib.graph_runtime.GraphModule(lib["default"](tvm.cpu(0)))

        runtime.set_input("input_ids", inp)

        runtime.run()

        tvm_outputs = []
        print("num outputs:", runtime.get_num_outputs())

        for i in range(runtime.get_num_outputs()):
            out = runtime.get_output(i)
            tvm_outputs.append(out.asnumpy())

        for pt_out, tvm_out in zip(torch_outputs, tvm_outputs):
            num_identical = np.sum(pt_out == tvm_out)
            match_ratio = num_identical / float(np.prod(tvm_out.shape))
            print(pt_out.shape, tvm_out.shape)
            print(
                np.max(np.abs(pt_out - tvm_out)),
                np.mean(np.abs(pt_out - tvm_out)),
                match_ratio,
            )


gpt2_test()
