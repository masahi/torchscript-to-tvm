import torch
import numpy as np
import tvm
from tvm import relay

model = torch.nn.Transformer(d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6)
model = model.eval()
src = torch.rand((10, 32, 256))
tgt = torch.rand((20, 32, 256))
input = [src, tgt]

trace = torch.jit.trace(model, input)
input_names = ["input{}".format(idx) for idx, inp in enumerate(input)]
input_shapes = list(zip(input_names, [inp.shape for inp in input]))

with torch.no_grad():
    pt_result = model(*input).numpy()

mod, params = relay.frontend.from_pytorch(trace, input_shapes)
print("imported")

target = "vulkan -from_device=0"

from tvm.relay.transform import InferType, ToMixedPrecision, mixed_precision
# mod = ToMixedPrecision("float16")(mod)

# with tvm.transform.PassContext(opt_level=3, disabled_pass=["FoldConstant"]):
#     opt_mod, opt_params = relay.optimize(mod, target=target, params=params)
#     print(opt_mod)

with tvm.transform.PassContext(opt_level=3, disabled_pass=["FoldConstant"]):
    json, lib, params = relay.build(mod, target=target, params=params)

ctx = tvm.device(target, 0)
runtime = tvm.contrib.graph_executor.create(json, lib, ctx)
runtime.set_input(**params)
runtime.set_input("input0", input[0].numpy())
runtime.set_input("input1", input[1].numpy())
runtime.run()

tvm_result = runtime.get_output(0).asnumpy()

print(np.mean(np.abs(tvm_result - pt_result)))
