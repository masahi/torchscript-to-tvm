import torch
import torchvision
import numpy as np
from tvm import relay
from tvm.relay.frontend.pytorch import from_pytorch, get_graph_input_names


def do_script(model, in_size=100):
    model_script = torch.jit.script(model)
    model_script.eval()
    return model_script


def do_trace(model, in_size=100):
    model_trace = torch.jit.trace(model, torch.rand(1, 3, in_size, in_size))
    model_trace.eval()
    return model_trace


def dict_to_tuple(out_dict):
    if "masks" in out_dict.keys():
        return (out_dict["boxes"], out_dict["scores"], out_dict["labels"], out_dict["masks"])
    return (out_dict["boxes"], out_dict["scores"], out_dict["labels"])


class TraceWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inp):
        out = self.model(inp)
        return dict_to_tuple(out[0])


def save_jit_model(script=False):
    model_funcs = [torchvision.models.detection.fasterrcnn_resnet50_fpn,
                   torchvision.models.detection.maskrcnn_resnet50_fpn]

    names = ["faster_rcnn", "mask_rcnn"]

    for name, model_func in zip(names, model_funcs):
        if script:
            model = model_func(num_classes=50, pretrained_backbone=False)
        else:
            model = TraceWrapper(model_func(num_classes=50, pretrained_backbone=False))

        model.eval()
        in_size = 100
        inp = torch.rand(1, 3, in_size, in_size)

        with torch.no_grad():
            out = model(inp)

            if script:
                out = dict_to_tuple(out[0])
                script_module = do_script(model)
                script_out = script_module([inp[0]])[1]
                script_out = dict_to_tuple(script_out[0])
            else:
                script_module = do_trace(model)
                script_out = script_module(inp)

            assert len(out[0]) > 0 and len(script_out[0]) > 0

            # compare bbox coord
            print(np.max(np.abs(out[0].numpy() - script_out[0].numpy())))

            torch._C._jit_pass_inline(script_module.graph)
            torch.jit.save(script_module, name + ".pt")


def convert_roi_align():
    def _impl(inputs, input_types):
        spatial_scale = inputs[2]
        pooled_size = (inputs[3], inputs[4])
        sampling_ratio = inputs[5]
        return relay.op.vision.roi_align(inputs[0], inputs[1],
                                         pooled_size, spatial_scale,
                                         sampling_ratio)
    return _impl


script_module = torch.jit.load("mask_rcnn.pt")
input_name = get_graph_input_names(script_module)[0]
input_shapes = {input_name: (1, 3, 300, 300)}
custom_map = {'torchvision::roi_align': convert_roi_align()}
from_pytorch(script_module, input_shapes, custom_map)
"""
NotImplementedError: The following operators are not implemented: ['aten::expand_as', 'aten::__and__', 'aten::meshgrid', 'aten::__interpolate', 'aten::scatter', 'aten::nonzero', 'aten::split_with_sizes', 'aten::log2', 'prim::ImplicitTensorToNum', 'aten::index', 'aten::_shape_as_tensor', 'aten::scalar_tensor']
"""
