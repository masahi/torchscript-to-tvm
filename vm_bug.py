import numpy as np
import torch
import tvm
from tvm import relay

from torch_frontend import parse_script_module


input_name = 'X'
input_shapes = {input_name: (10, 20)}


class SimpleLoopVM_bug(torch.nn.Module):
    def forward(self, inp):
        a = torch.zeros((10, 20))
        for i in range(inp.size(0)):
            a += inp
        return a


models = [
    SimpleLoopVM_bug().eval(),
]

for raw_model in models:
    script_module = torch.jit.script(raw_model)
    mod, params = parse_script_module(script_module, input_shapes)
    print(mod["main"])

    executor = relay.create_executor("vm", mod=mod, ctx=tvm.cpu(0), target="llvm")
    evaluator = executor.evaluate()

    inp = torch.rand(input_shapes[input_name], dtype=torch.float)
    expected = inp.numpy() * inp.shape[0]

    params[input_name] = inp.numpy()
    tvm_res = evaluator(**params)

    tvm.testing.assert_allclose(tvm_res.asnumpy(), expected, rtol=1e-5, atol=1e-5)
