import os
import numpy as np
import torch
import tvm
from tvm import relay
from torchvision import models

from torch_frontend import parse_script_module

inp = torch.rand(1, 3, 224, 224, dtype=torch.float)
input_name = 'X'
input_shapes = {input_name: (1, 3, 224, 224)}

models = [
    models.resnet.resnet18(pretrained=True).eval(),
    models.vgg.vgg16_bn(pretrained=True).eval(),
    models.mobilenet.mobilenet_v2(pretrained=True).eval(),
    models.squeezenet.squeezenet1_1(pretrained=True).eval(),
    models.densenet.densenet121(pretrained=True).eval(),
    models.inception.inception_v3(pretrained=True).eval(),
    models.mnasnet.mnasnet1_0(pretrained=True).eval()
]

for raw_model in models:
    script_module = torch.jit.trace(raw_model, inp).eval()
    mod, params = parse_script_module(script_module, input_shapes)

    with torch.no_grad():
        pt_result = raw_model(inp).numpy()

    with relay.build_config(opt_level=3):
        json, lib, params = relay.build(mod, target="llvm", params=params)

    runtime = tvm.contrib.graph_runtime.create(json, lib, tvm.context("cpu", 0))
    runtime.set_input(**params)
    runtime.set_input("X", inp.numpy())
    runtime.run()
    tvm_result = runtime.get_output(0).asnumpy()
    np.allclose(tvm_result, pt_result, rtol=1e-5, atol=1e-5)
    print(np.max(np.abs(tvm_result - pt_result)))
