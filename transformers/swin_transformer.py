import numpy as np
import tvm
import torch
from tvm import relay
from swin_transformer import SwinTransformer

net = SwinTransformer().eval()

img = torch.randn(1, 3, 224, 224)

scripted_model = torch.jit.trace(net, img).eval()
input_name = "input0"
shape_list = [(input_name, img.shape)]
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

with torch.no_grad():
    pt_result = net(img).numpy()

target = "llvm"

with tvm.transform.PassContext(opt_level=3):
    json, lib, params = relay.build(mod, target=target, params=params)

ctx = tvm.device(target, 0)
runtime = tvm.contrib.graph_executor.create(json, lib, ctx)
runtime.set_input(**params)
runtime.set_input("input0", img.numpy())
runtime.run()

tvm_result = runtime.get_output(0).asnumpy()

print(np.mean(np.abs(tvm_result - pt_result)))
