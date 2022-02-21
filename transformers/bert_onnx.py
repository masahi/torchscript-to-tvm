import os
import numpy as np
import torch

from transformers import BertForSequenceClassification

import tvm
from tvm import relay

import onnx


def export(model, onnx_model_path):
    with torch.no_grad():
        inputs = {
            "input_ids": torch.ones(1, 128, dtype=torch.int64),
            "attention_mask": torch.ones(1, 128, dtype=torch.int64),
            "token_type_ids": torch.ones(1, 128, dtype=torch.int64),
        }

        with torch.no_grad():
            model(*inputs.values())

        symbolic_names = {0: "batch_size", 1: "max_seq_len"}

        torch.onnx.export(
            model,  # model being run
            (
                inputs["input_ids"],  # model input (or a tuple for multiple inputs)
                inputs["attention_mask"],
                inputs["token_type_ids"],
            ),  # model input (or a tuple for multiple inputs)
            onnx_model_path,  # where to save the model (can be a file or file-like object)
            opset_version=11,  # the ONNX version to export the model to
            do_constant_folding=True,  # whether to execute constant folding for optimization
            input_names=[
                "input_ids",  # the model's input names
                "attention_mask",
                "token_type_ids",
            ],
            output_names=["output"],  # the model's output names
            dynamic_axes={
                "input_ids": symbolic_names,  # variable length axes
                "attention_mask": symbolic_names,
                "token_type_ids": symbolic_names,
            },
        )


onnx_path = "bert-large.onnx"

if not os.path.exists(onnx_path):
    model = BertForSequenceClassification.from_pretrained(
        "bert-large-uncased", return_dict=False
    )
    export(model, onnx_path)


onnx_model = onnx.load(onnx_path)
batch_size = 1
seq_len = 128

shape_dict = {
    "input_ids": (batch_size, seq_len),
    "attention_mask": (batch_size, seq_len),
    "token_type_ids": (batch_size, seq_len),
}

if not os.path.exists("bert_large_onnx.json"):
    import time

    t1 = time.time()
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)
    mod = relay.transform.DynamicToStatic()(mod)
    t2 = time.time()

    print(relay.transform.InferType()(mod))

    print("ONNX import time:", t2 - t1)


    with open("bert_large_onnx.json", "w") as fo:
        fo.write(tvm.ir.save_json(mod))

    with open("bert_large_onnx.params", "wb") as fo:
        fo.write(relay.save_param_dict(params))

with open("bert_large_onnx.json", "r") as fi:
    mod = tvm.ir.load_json(fi.read())

with open("bert_large_onnx.params", "rb") as fi:
    params = relay.load_param_dict(fi.read())

target = "llvm -mcpu=cascadelake"
with tvm.transform.PassContext(opt_level=3):
    opt_mod, opt_params = relay.optimize(mod, target=target, params=params)
    print(opt_mod["main"])
