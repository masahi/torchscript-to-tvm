from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

import tvm
from tvm import relay
from tvm.relay.frontend.pytorch import from_pytorch
from tvm.relay.ty import TupleType, TensorType
from tvm.relay.prelude import Prelude


def vmobj_to_list(o, dtype="float32"):
    if isinstance(o, tvm.nd.NDArray):
        return [o]
    elif isinstance(o, tvm.runtime.container.ADT):
        result = []
        for f in o:
            result.extend(vmobj_to_list(f, dtype))
        return result
    else:
        raise RuntimeError("Unknown object type: %s" % type(o))


def assert_equal(tvm_result, torch_result):
    if isinstance(torch_result, (tuple, list)):
        assert isinstance(tvm_result, list)
        for tvm_res, pt_res in zip(tvm_result, torch_result):
            assert_equal(tvm_res, pt_res)
    elif isinstance(torch_result, torch.Tensor):
        print(np.max(np.abs(tvm_result.asnumpy() - torch_result.numpy())))
        tvm.testing.assert_allclose(tvm_result.asnumpy(), torch_result.numpy(),
                                    rtol=1e-5, atol=1e-5)
    else:
        tvm_res = np.asscalar(tvm_result.asnumpy())
        print(abs(torch_result - tvm_res))
        assert torch_result == tvm_res


def run_and_compare(mod, params, pt_result):
    executor = relay.create_executor("vm", mod=mod, ctx=tvm.cpu(0), target="llvm")
    evaluator = executor.evaluate()

    exec_res = evaluator(**params)

    def flatten(nested):
        res = []
        for r in nested:
            if isinstance(r, torch.Tensor):
                res.append(r)
            else:
                res.extend(flatten(r))
        return res

    if isinstance(exec_res, tvm.runtime.container.ADT):
        assert not isinstance(pt_result, torch.Tensor)
        tvm_res = vmobj_to_list(exec_res)
        torch_res = flatten(pt_result)
    else:
        tvm_res = exec_res
        torch_res = pt_result

    assert_equal(tvm_res, torch_res)


def simple_rnn_test():
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
        def __init__(self):
            super().__init__()
            x = torch.rand(10, 4, dtype=torch.float)
            h = torch.rand(10, 4, dtype=torch.float)
            self.cell = torch.jit.trace(Cell(DecisionGate()), (x, h))

        def forward(self, xs):
            h = torch.zeros(10, 4, dtype=torch.float)
            y = torch.zeros(10, 4, dtype=torch.float)
            for i in range(xs.size(0)):
                y, h = self.cell(xs[i], h)
            return y

    raw_model = RNNLoop()
    script_module = torch.jit.script(raw_model)
    input_name = "input"
    input_shapes = [(input_name, (10, 10, 4))]

    mod, params = from_pytorch(script_module, input_shapes, {})

    inp = torch.randn(input_shapes[0][1], dtype=torch.float)
    with torch.no_grad():
        pt_result = raw_model(inp.clone())

    params[input_name] = inp.numpy()

    run_and_compare(mod, params, pt_result)


class SimpleList(nn.Module):
    def forward(self, tensor, states):
        # type: (Tensor, List[Tensor]) -> Tensor
        return states[0]


class ListHead(nn.Module):
    def forward(self, tensor):
        # type: (Tensor) -> Tensor
        lst = [tensor]
        return lst[0]


class ListIdentity(nn.Module):
    def forward(self, tensor, states):
        # type: (Tensor, List[Tensor]) -> List[Tensor]
        return states


def convert_to_list_adt(py_lst, prelude):
    adt_lst = prelude.nil()
    for elem in reversed(py_lst):
        if isinstance(elem, np.ndarray):
            relay_val = relay.const(elem)
        elif isinstance(elem, tuple):
            relay_val = relay.Tuple([relay.const(e) for e in elem])
        adt_lst = prelude.cons(relay_val, adt_lst)
    return adt_lst


def convert_list_to_vmobj(py_list):
    mod = tvm.IRModule()
    prelude = Prelude(mod)
    adt_list = convert_to_list_adt(py_list, prelude)

    mod["main"] = relay.Function([], adt_list)
    intrp = relay.create_executor("vm", mod=mod, ctx=tvm.cpu(0), target="llvm")
    adt_obj = intrp.evaluate(adt_list)
    return adt_obj


def custom_lstm_test():
    input_name = "input"
    states_name = "states"
    seq_len = 5
    batch = 2
    input_size = 3
    hidden_size = 4
    num_layers = 1

    input_shapes = [(input_name, (seq_len, batch, input_size)),
                    (states_name, ((batch, hidden_size), (batch, hidden_size)))]

    input_shapes_stacked = [(input_name, (seq_len, batch, input_size)),
                            (states_name, [((batch, hidden_size), (batch, hidden_size)),
                                           ((batch, hidden_size), (batch, hidden_size))])]

    tensor_list_shape = [(input_name, (seq_len, batch, input_size)),
                         (states_name, [(batch, hidden_size), (batch, hidden_size)])]
    # tensor_list_shape = [(input_name, (batch, hidden_size)),
    #                      (states_name, [(batch, hidden_size), (batch, hidden_size)])]

    state_list = [torch.rand(shape) for shape in tensor_list_shape[1][1]]

    inp = torch.randn(seq_len, batch, input_size)
    # inp = torch.randn(batch, hidden_size)

    states = [(torch.randn(batch, hidden_size),
               torch.randn(batch, hidden_size))
              for _ in range(num_layers)]

    from custom_lstms import lstmln_layer, stacked_rnn

    models = [
      (ListIdentity(), state_list, tensor_list_shape),
      (ListHead(), None, [("input", (10, 10))]),
      (SimpleList(), state_list, tensor_list_shape),
      (lstmln_layer(input_size, hidden_size).eval(), states[0], input_shapes),
      (stacked_rnn(input_size, hidden_size, num_layers).eval(), states, input_shapes_stacked)
    ]

    for (raw_model, states, input_shapes) in models:
        script_module = torch.jit.script(raw_model)
        mod, params = from_pytorch(script_module, input_shapes)
        print(mod["main"])

        with torch.no_grad():
            if states is None:
                pt_result = raw_model(inp.clone())
            else:
                pt_result = raw_model(inp.clone(), states)

        params[input_name] = inp.numpy()

        if states:
            if isinstance(states, tuple):
                states_np = tuple(st.numpy() for st in states)
            elif isinstance(states, list) and isinstance(states[0], torch.Tensor):
                states_np = [st.numpy() for st in states]
            elif isinstance(states, list) and isinstance(states[0], tuple):
                states_np = [tuple(st.numpy() for st in states[i])
                             for i in range(num_layers)]
            else:
                assert False

            if isinstance(states_np, list):
                params[states_name] = convert_list_to_vmobj(states_np)
            else:
                params[states_name] = states_np

        run_and_compare(mod, params, pt_result)


custom_lstm_test()
simple_rnn_test()
