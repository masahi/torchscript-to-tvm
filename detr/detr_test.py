import io
import numpy as np

import onnx
import torch

import sys
sys.path.append("../../../ml/detr/")

from hubconf import detr_resnet50

import tvm
from tvm import relay
import onnxruntime


class TraceWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inp):
        out = self.model(inp)
        return (out['pred_logits'], out['pred_boxes'])


def export_onnx(model, output_path, inp, output_names=None, input_names=None):
    torch.onnx.export(model, inp, output_path,
                      do_constant_folding=True, opset_version=12,
                      dynamic_axes=None, input_names=input_names, output_names=output_names)


def get_torch_outputs(model, inp):
    with torch.no_grad():
        raw_outputs = model(inp)
        outputs, _ = torch.jit._flatten(raw_outputs)
        return [output.cpu().numpy() for output in outputs]


def ort_validate(onnx_model, inputs, outputs):
    ort_session = onnxruntime.InferenceSession(onnx_model)
    # compute onnxruntime output prediction
    ort_inputs = dict((ort_session.get_inputs()[i].name, inpt) for i, inpt in enumerate(inputs))
    ort_outs = ort_session.run(None, ort_inputs)
    for i in range(0, len(outputs)):
        torch.testing.assert_allclose(outputs[i], ort_outs[i], rtol=1e-03, atol=1e-05)


def test_model_onnx_detection():
    model = detr_resnet50(pretrained=False).eval()
    inp = torch.rand(1, 3, 750, 800)

    onnx_io = io.BytesIO()

    export_onnx(
        model,
        onnx_io,
        inp,
        input_names=["inputs"],
        output_names=["pred_logits", "pred_boxes"]
    )

    inputs = [inp.numpy()]
    outputs = get_torch_outputs(model, inp)

    ort_validate(onnx_io.getvalue(), inputs, outputs)


def test_load_tvm():
    model = detr_resnet50(pretrained=False).eval()
    inp = torch.rand(1, 3, 750, 800)

    onnx_path = "detr.onnx"

    # export_onnx(
    #     model,
    #     onnx_path,
    #     inp,
    #     input_names=["inputs"],
    #     output_names=["pred_logits", "pred_boxes"]
    # )

    # onnx_model = onnx.load(onnx_path)

    input_name = "inputs"
    shape_dict = {input_name: inp.shape}
    # tvm.error.OpNotImplemented: The following operators are not supported for frontend ONNX: CumSum
    # mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

    model = TraceWrapper(model)

    with torch.no_grad():
        trace = torch.jit.trace(model, inp)

    mod, params = relay.frontend.from_pytorch(trace, [('input', inp.shape)])

    target = "cuda"
    with tvm.transform.PassContext(opt_level=3, disabled_pass=["FoldConstant", "DynamicToStatic"]):
        json, lib, params = relay.build(mod, target=target, params=params)

    ctx = tvm.context(target, 0)
    runtime = tvm.contrib.graph_runtime.create(json, lib, ctx)
    runtime.set_input(**params)
    runtime.set_input("input", inp.numpy())
    runtime.run()

    tvm_results = [runtime.get_output(i).asnumpy() for i in [0, 1]]
    pt_results = get_torch_outputs(model, inp)

    for pt_res, tvm_res in zip(pt_results, tvm_results):
        print(np.mean(np.abs(pt_res - tvm_res)))


# test_model_onnx_detection()
test_load_tvm()
