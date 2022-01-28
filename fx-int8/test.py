import torch
import torchvision
import numpy as np
from torch.quantization import get_default_qconfig
from torch.quantization.quantize_fx import prepare_fx, convert_fx
import cv2
from tvm.contrib.download import download
from tvm import relay


def get_input(in_size):
    img_path = "test_street_small.jpg"
    img_url = (
        "https://raw.githubusercontent.com/dmlc/web-data/" "master/gluoncv/detection/street_small.jpg"
    )
    download(img_url, img_path)

    img = cv2.imread(img_path).astype("float32")
    img = cv2.resize(img, (in_size, in_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img / 255.0, [2, 0, 1])
    img = np.expand_dims(img, axis=0)
    return img


def do_trace(model, in_size=500):
    model_trace = torch.jit.trace(model, torch.rand(1, 3, in_size, in_size))
    model_trace.eval()
    return model_trace


def dict_to_tuple(out_dict):
    if "masks" in out_dict.keys():
        return out_dict["boxes"], out_dict["scores"], out_dict["labels"], out_dict["masks"]
    return out_dict["boxes"], out_dict["scores"], out_dict["labels"]


def quantize(model_fp):
    qconfig = get_default_qconfig("fbgemm")
    qconfig_dict = {"": qconfig}
    # model_fp = quantize_fx.fuse_fx(model_fp)
    return convert_fx(prepare_fx(model_fp, qconfig_dict))


def test_ssd_vgg():
    class TraceWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, inp):
            features = self.model.backbone(inp)
            features = list(features.values())
            out = self.model.head(features)
            return out["bbox_regression"], out["cls_logits"]

    model_func = torchvision.models.detection.ssd300_vgg16
    model = TraceWrapper(model_func(num_classes=50, pretrained_backbone=True)).eval()

    model = quantize(model)

    in_size = 500
    img = get_input(in_size)
    inp = torch.from_numpy(img)
    input_name = "inp"

    with torch.no_grad():
        script_module = do_trace(model, in_size)
        mod, params = relay.frontend.from_pytorch(script_module, [(input_name, inp.shape)])

    print(relay.transform.InferType()(mod))


def test_deeplab_v3():
    class TraceWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, inp):
            out = self.model(inp)
            return out["out"]

    deeplabv3 = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True)
    model = TraceWrapper(deeplabv3.eval()).eval()
    inp = torch.rand(8, 3, 512, 512)

    qmodel = quantize(model)

    with torch.no_grad():
        trace = torch.jit.trace(qmodel, inp)
        torch_res = model(inp)

    mod, params = relay.frontend.from_pytorch(trace, [('input', inp.shape)])
    print(relay.transform.InferType()(mod))


def test_yolov5():
    class TraceWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, inp):
            out = self.model(inp)
            return dict_to_tuple(out[0])

    from yolort.models import yolov5l

    in_size = 500
    model = yolov5l(export_friendly=True, pretrained=True)
    model.eval()
    inp = torch.Tensor(np.random.uniform(0.0, 250.0, size=(1, 3, in_size, in_size)))

    qmodel = model
    qmodel.model.backbone =  quantize(model.model.backbone)
    # qmodel.model.head.head = quantize(model.model.head.head)

    model = TraceWrapper(qmodel)

    with torch.no_grad():
        out = model(inp)
        model_trace = torch.jit.trace(model, inp).eval()

    mod, params = relay.frontend.from_pytorch(model_trace, [('input', inp.shape)])

    print(relay.transform.InferType()(mod))


def test_imagenet():
    from torchvision.models.efficientnet import efficientnet_b4
    from torchvision.models.resnet import resnet50

    for model_func in [resnet50, efficientnet_b4]:
        model = efficientnet_b4(pretrained=True).eval()
        model = quantize(model)

        x = torch.rand((1, 3, 224, 224))
        model_traced = torch.jit.trace(model, x).eval()

        mod, params = relay.frontend.from_pytorch(model_traced, [("x", x.shape)])
        print(relay.transform.InferType()(mod))


test_ssd_vgg()
test_deeplab_v3()
test_yolov5()
test_imagenet()
