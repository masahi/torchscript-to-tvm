import numpy as np
import torch
import tvm
from tvm import relay
from tvm.relay.backend import vm
from tvm.relay import TupleType, TensorType
from torch_frontend import parse_script_module
import itertools

from custom_lstms import rnn_layer, stacked_rnn, stacked_lnlstm


class DecisionGate(torch.nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return x
        else:
            return -x


class Cell(torch.nn.Module):
    def __init__(self, dg):
        super(Cell, self).__init__()
        self.dg = dg
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.dg(self.linear(x)) + h)
        return new_h, new_h


class RNNLoop(torch.nn.Module):
    def __init__(self, scripted_gate):
        super().__init__()
        x = torch.rand(10, 4, dtype=torch.float)
        h = torch.rand(10, 4, dtype=torch.float)
        self.cell = torch.jit.trace(Cell(scripted_gate), (x, h))

    def forward(self, xs):
        h, y = torch.zeros(10, 4, dtype=torch.float), torch.zeros(10, 4, dtype=torch.float)
        for i in range(xs.size(0)):
            y, h = self.cell(xs[i], h)
        return y


seq_len = 5
batch = 2
input_size = 3
hidden_size = 4
num_layers = 4

input_name = 'X'
# input_shapes = {input_name: (10, 10, 4)}
input_shapes = {}
input_types = {input_name: TensorType((seq_len, batch, input_size)),
               "states": TupleType([TensorType((batch, hidden_size)),
                                    TensorType((batch, hidden_size))])}

gate = DecisionGate()

models = [
    rnn_layer(input_size, hidden_size).eval(),
#    stacked_rnn(input_size, hidden_size, num_layers).eval(),
#    stacked_lnlstm(input_size, hidden_size, num_layers).eval()
]
"""
Missing conversion
['aten::__getitem__', 'aten::layer_norm']
"""
for raw_model in models:
    script_module = torch.jit.script(raw_model)
    break
    mod, params = parse_script_module(script_module, input_shapes, input_types)
    print(mod)
    continue

    for k, v in params.items():
        print(k, v.shape)

    # comp = vm.VMCompiler()
    # opt_mod, _ = comp.optimize(mod, "llvm", params)
    # print(opt_mod)

    executor = relay.create_executor("vm", mod=mod, ctx=tvm.cpu(0), target="llvm")
    evaluator = executor.evaluate()

    for i in range(5):
        inp = torch.randn(seq_len, batch, input_size)
        states = [(torch.randn(batch, hidden_size), torch.randn(batch, hidden_size))
                  for _ in range(num_layers)]

        with torch.no_grad():
            pt_result = raw_model(inp.clone(), states[0])

        params[input_name] = inp.numpy()
        params["states"] = (st.numpy() for st in states[0])
        op_res = evaluator(**params)

        if not isinstance(pt_result, torch.Tensor):
            tvm_res = np.asscalar(op_res.asnumpy())
            print(abs(pt_result - tvm_res))
            assert pt_result == tvm_res
        else:
            print(np.max(np.abs(op_res.asnumpy() - pt_result.numpy())))
            tvm.testing.assert_allclose(op_res.asnumpy(), pt_result.numpy(),
                                        rtol=1e-5, atol=1e-5)

graph = script_module.graph
list_construct_ops = graph.findAllNodes("prim::ListConstruct")
tensor_list_ops = [op for op in list_construct_ops if str(op.output().type()) == "List[Tensor]"]


def get_use_chains(root_node):
    def concat_lists(lists):
        return itertools.chain.from_iterable(lists)

    def inner(current, accum):
        users = []
        for output in current.outputs():
            users += [use.user for use in output.uses()]

        if not users:
            return [accum]

        return concat_lists([inner(nxt, accum + [nxt]) for nxt in users])

    return inner(root_node, [root_node])


def has_kind(chain, kind):
    return any([node.kind() == kind for node in chain])


def get_node(node_list, kind, filter_func=lambda node: True):
    for node in node_list:
        if node.kind() == kind and filter_func(node):
            return node
    assert False
    return None


chains = []
for tensor_list_op in tensor_list_ops:
    chains += get_use_chains(tensor_list_op)

chains = [chain for chain in chains
          if has_kind(chain, "aten::stack") and has_kind(chain, "prim::Loop")]

for chain in chains:
    tensor_list_op = chain[0]
    loop_op = get_node(chain, "prim::Loop")

    tarray_create_node = graph.create("relay::tensor_array_create")
    tarray_create_node.insertBefore(loop_op)
    tensor_list_op.replaceAllUsesWith(tarray_create_node)
    tensor_list_op.destroy()

    stack_op = get_node(chain, "aten::stack")
    tarray_stack_node = graph.create("relay::tensor_array_stack", [loop_op.outputsAt(0)])
    tarray_stack_node.insertBefore(stack_op)
    stack_op.replaceAllUsesWith(tarray_stack_node)
    stack_op.destroy()

    loop_block = list(loop_op.blocks())[0]
    loop_nodes = list(loop_block.nodes())

    list_add_op = get_node(loop_nodes, "aten::add_",
                           lambda node: str(node.output().type()) == "List[Tensor]")

    list_singlton_op = list_add_op.inputsAt(1).node()
    list_singlton_op_input = list_singlton_op.inputsAt(0)
    list_singlton_op.output().replaceAllUsesWith(list_singlton_op_input)
    list_singlton_op.destroy()

    tarray_write_node = graph.create("relay::tensor_array_write", list(list_add_op.inputs()))
    tarray_write_node.insertBefore(list_add_op)
    list_add_op.replaceAllUsesWith(tarray_write_node)
    list_add_op.destroy()


#torch._C._jit_pass_dce(graph)
#torch._C._jit_pass_inline(graph)
print(graph)
# for use in outputs1.output().uses():
#     print(use.user)
