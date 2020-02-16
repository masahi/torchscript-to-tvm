import numpy as np
import torch
import tvm
from tvm import relay
from torchvision import models

from torch_frontend import parse_script_module


class SegmentationModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inp):
        out = self.model(inp)
        return out["out"]


def dict_to_tuple(out_dict):
    if "masks" in out_dict.keys():
        return (out_dict["boxes"], out_dict["scores"], out_dict["labels"], out_dict["masks"])
    return (out_dict["boxes"], out_dict["scores"], out_dict["labels"])


class DetectionModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inp):
        out = self.model(inp)
        return dict_to_tuple(out[0])


def run_on_models(models, inp, input_shapes, target="llvm"):
    for raw_model in models:
        script_module = torch.jit.trace(raw_model, inp).eval()
        mod, params = parse_script_module(script_module, input_shapes)

        with torch.no_grad():
            pt_result = raw_model(inp).numpy()

        with relay.build_config(opt_level=3):
            json, lib, params = relay.build(mod, target=target, params=params)

        ctx = tvm.context(target, 0)
        runtime = tvm.contrib.graph_runtime.create(json, lib, ctx)
        runtime.set_input(**params)
        runtime.set_input("X", inp.numpy())
        runtime.run()

        tvm_result = runtime.get_output(0).asnumpy()
        np.allclose(tvm_result, pt_result, rtol=1e-5, atol=1e-5)
        print(np.max(np.abs(tvm_result - pt_result)),
              np.mean(np.abs(tvm_result - pt_result)))

        tvm.testing.assert_allclose(tvm_result, pt_result,
                                    rtol=1e-3, atol=1e-3)


def imagenet_test():
    inp = torch.rand(1, 3, 224, 224, dtype=torch.float)
    input_name = 'X'
    input_shapes = {input_name: (1, 3, 224, 224)}

    test_models = [
        # models.resnet.resnet18(pretrained=True).eval(),
        models.mobilenet.mobilenet_v2(pretrained=True).eval(),
        # models.squeezenet.squeezenet1_1(pretrained=True).eval(),
        # models.densenet.densenet121(pretrained=True).eval(),
        # models.inception.inception_v3(pretrained=True).eval(),
        # models.mnasnet.mnasnet1_0(pretrained=True).eval(),
        # models.alexnet(pretrained=True).eval(),
        # models.vgg.vgg11_bn(pretrained=True).eval(),
    ]

    for target in ["llvm"]:
        run_on_models(test_models, inp, input_shapes, target)


def segmentation_test():
    input_name = 'X'
    input_shapes = {input_name: (1, 3, 300, 300)}
    inp = torch.rand(input_shapes[input_name], dtype=torch.float)

    fcn = models.segmentation.fcn_resnet101(pretrained=True).eval()
    deeplab = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()

    test_models = [
       SegmentationModelWrapper(fcn),
       SegmentationModelWrapper(deeplab),
    ]

    for target in ["llvm", "cuda"]:
        run_on_models(test_models, inp, input_shapes, target)


def detection_test():
    input_name = 'X'
    input_shapes = {input_name: (1, 3, 100, 100)}
    inp = torch.rand(input_shapes[input_name], dtype=torch.float)

    test_models = []
    for model_func in [models.detection.fasterrcnn_resnet50_fpn,
                       models.detection.maskrcnn_resnet50_fpn]:
        detection_model = model_func(num_classes=50, pretrained_backbone=False)
        test_models.append(DetectionModelWrapper(detection_model))

    for target in ["llvm", "cuda"]:
        run_on_models(test_models, inp, input_shapes, target)


# imagenet_test()
# segmentation_test()
detection_test()
