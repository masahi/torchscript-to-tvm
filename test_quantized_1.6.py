import numpy as np
import torch.onnx
import torch.nn as nn
import torch.nn.quantized as nnq
import io

import onnx
from tvm import relay


def generic_test(model, sample_inputs, input_names=None, decimal=3, relaxed_check=False):
    torch.backends.quantized.engine = "qnnpack"
    pt_inputs = tuple(torch.from_numpy(x) for x in sample_inputs)
    model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    q_model = torch.quantization.prepare(model, inplace=False)
    q_model = torch.quantization.convert(q_model, inplace=False)

    traced_model = torch.jit.trace(q_model, pt_inputs)
    buf = io.BytesIO()
    torch.jit.save(traced_model, buf)
    buf.seek(0)
    q_model = torch.jit.load(buf)

    q_model.eval()
    output = q_model(*pt_inputs)

    f = io.BytesIO()
    torch.onnx.export(q_model, pt_inputs, f, input_names=input_names, example_outputs=output,
                      operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)


def export_to_onnx(model, input, input_names):
    torch.backends.quantized.engine = "fbgemm"
    qconfig = torch.quantization.default_qconfig
    model.qconfig = qconfig
    model = torch.quantization.prepare(model)
    model = torch.quantization.convert(model)

    outputs = model(input)

    traced = torch.jit.trace(model, input)

    mod, params = relay.frontend.from_pytorch(traced, [("input", input.shape)])
    print(mod["main"])

    buf = io.BytesIO()
    torch.jit.save(traced, buf)
    buf.seek(0)

    model = torch.jit.load(buf)
    f = io.BytesIO()
    torch._C._jit_pass_inline(model.graph)
    # print(model.graph)
    # print(traced.state_dict())

    torch.onnx.export(model, input, f, input_names=input_names, example_outputs=outputs,
                      operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    f.seek(0)

    onnx_model = onnx.load(f)
    return onnx_model


def test_qconv_model():
    class ConvModel(torch.nn.Module):
        def __init__(self):
            super(ConvModel, self).__init__()
            self.qconfig = torch.quantization.default_qconfig
            self.fc1 = torch.quantization.QuantWrapper(torch.nn.Conv2d(3, 5, 2, bias=True).to(dtype=torch.float))

        def forward(self, x):
            x = self.fc1(x)
            return x

    model = ConvModel()
    # weight, bias = torch.ops.quantized.conv2d_unpack(model.fc1.module._packed_params)
    # print(weight.shape, bias.shape)

    x_numpy = np.random.rand(1, 3, 6, 6).astype(np.float32)
    x = torch.from_numpy(x_numpy).to(dtype=torch.float)
    input_names = ["x"]
    export_to_onnx(model, x, input_names)


def test_small_model():
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.quant = torch.quantization.QuantStub()
            self.dequant = torch.quantization.DeQuantStub()
            self.func_add = nnq.FloatFunctional()
            self.conv1 = nn.Conv2d(3, 2, 5, bias=None).to(dtype=torch.float)
            self.act1 = nn.Sigmoid()
            self.conv2 = nn.Conv2d(2, 2, 1, bias=None).to(dtype=torch.float)
            self.fc = nn.Linear(72, 10).to(dtype=torch.float)
            self.fc.qconfig = None

        def forward(self, x):
            x = self.quant(x)
            x = self.func_add.add(x, x)
            x = self.conv1(x)
            x = self.act1(x)
            x = self.conv2(x)
            x = self.dequant(x)
            x = x.reshape(-1, 72).contiguous()
            x = self.fc(x)
            return x

    x = torch.from_numpy(np.random.rand(2, 3, 10, 10).astype("float32"))
    export_to_onnx(SimpleModel(), x, ["x"])


def test_sequential():
    class ConvBNReLUModule(nn.Sequential):
        def __init__(self):
            super().__init__(
                nn.Conv2d(3, 3, 1, 1, bias=False),
                nn.BatchNorm2d(3),
                nn.ReLU(inplace=False)
            )

    class ModelWithClassifierHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 3, 1)
            self.relu1 = nn.ReLU(inplace=False)
            layers = []
            for i in range(3):
                layers.append(ConvBNReLUModule())
            self.features = nn.Sequential(*layers)
            head = [nn.Linear(300, 10), nn.ReLU(inplace=False)]
            self.classifier = nn.Sequential(*head)
            self.seq = nn.Sequential()
            self.quant = torch.quantization.QuantStub()
            self.dequant = torch.quantization.DeQuantStub()

        def forward(self, x):
            x = self.quant(x)
            x = self.conv1(x)
            x = self.relu1(x)
            x = self.features(x)
            x = torch.reshape(x, (-1, 3 * 10 * 10))
            x = self.classifier(x)
            x = self.seq(x)
            x = self.dequant(x)
            return x

    model = ModelWithClassifierHead().eval()
    torch.quantization.fuse_modules(model, [['conv1', 'relu1'] ,
                                            ['features.0.0', 'features.0.1', 'features.0.2'],
                                            ['features.1.0', 'features.1.1', 'features.1.2'],
                                            ['features.2.0', 'features.2.1', 'features.2.2']], inplace=True)


    x = np.random.rand(1, 3, 10, 10).astype("float32")
    generic_test(model, (x,), input_names=["x"], relaxed_check=True)


test_qconv_model()
# test_small_model()
