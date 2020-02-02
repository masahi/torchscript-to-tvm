import torch
import torchvision


def do_script(model):
    model_script = torch.jit.script(model)
    model_script.eval()

    # compute predictions
    # predictions = model_script([torch.rand(3, 300, 300)])

    torch._C._jit_pass_inline(model_script.graph)

    print(model_script.graph)


# trace not working yet
def do_trace(model):
    model.eval()
    model_trace = torch.jit.trace(model, torch.rand(1, 3, 300, 300))
    model_trace.eval()

    # compute predictions
    # predictions = model_trace([torch.rand(3, 300, 300)])

    # torch._C._jit_pass_inline(model_trace.graph)

    print(model_trace.graph)


model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
do_script(model)
