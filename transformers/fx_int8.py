import numpy as np
import torch

from transformers import BertForSequenceClassification
from torch.quantization import get_default_qconfig
from torch.quantization.quantize_fx import prepare_fx, convert_fx

import tvm
from tvm import relay

model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

batch_size = 1
seq_len = 128
inputs = (torch.ones(batch_size, 64, dtype=torch.int64),
          torch.ones(batch_size, 64, dtype=torch.int64),
          torch.ones(batch_size, 64, dtype=torch.int64))

qconfig = get_default_qconfig("fbgemm")
qconfig_dict = {"": qconfig}

prepared_model = prepare_fx(model, qconfig_dict)
prepared_model(inputs)
qmodel = convert_fx(prepared_model)
print(qmodel.graph)

input_shapes = [("input_ids", (inputs[0].shape, "int64")),
                ("attention_mask", (inputs[1].shape, "int64")),
                ("token_type_ids", (inputs[2].shape, "int64"))]

with torch.no_grad():
    out = qmodel(*inputs)

class TraceWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *inp):
        out = self.model(*inp)
        return out["logits"]


script_module = torch.jit.trace(TraceWrapper(qmodel), inputs).eval()
mod, params = relay.frontend.from_pytorch(script_module, input_shapes)

target = "llvm"

with tvm.transform.PassContext(opt_level=3):
    # opt_mod, opt_params = relay.optimize(mod, target=target, params=params)
    # print(opt_mod["main"])
    lib = relay.build(mod, target=target, params=params)

runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](tvm.device(target, 0)))

runtime.set_input("input_ids", inputs[0].numpy())
runtime.set_input("attention_mask", inputs[1].numpy())
runtime.set_input("token_type_ids", inputs[2].numpy())

runtime.run()
