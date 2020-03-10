import numpy as np
import torch
import tvm
from tvm import relay
from tvm.relay import TupleType, TensorType
from tvm.relay.frontend.pytorch import from_pytorch, get_graph_input_names


def run_and_compare(mod, params, pt_result):
    executor = relay.create_executor("vm", mod=mod, ctx=tvm.cpu(0), target="llvm")
    evaluator = executor.evaluate()

    op_res = evaluator(**params)

    if not isinstance(pt_result, torch.Tensor):
        tvm_res = np.asscalar(op_res.asnumpy())
        print(abs(pt_result - tvm_res))
        assert pt_result == tvm_res
    else:
        print(np.max(np.abs(op_res.asnumpy() - pt_result.numpy())))
        tvm.testing.assert_allclose(op_res.asnumpy(), pt_result.numpy(),
                                    rtol=1e-5, atol=1e-5)


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
    input_name = get_graph_input_names(script_module)[0]
    input_shapes = {input_name: (10, 10, 4)}

    mod, params = from_pytorch(script_module, input_shapes, {})

    for i in range(5):
        inp = torch.randn(input_shapes[input_name], dtype=torch.float)
        with torch.no_grad():
            pt_result = raw_model(inp.clone())

        params[input_name] = inp.numpy()

        run_and_compare(mod, params, pt_result)


def custom_lstm_test():
    input_name = 'X'
    seq_len = 5
    batch = 2
    input_size = 3
    hidden_size = 4
    num_layers = 4

    input_shapes = {}
    input_types = {input_name: TensorType((seq_len, batch, input_size)),
                   "states": TupleType([TensorType((batch, hidden_size)),
                                        TensorType((batch, hidden_size))])}

    from custom_lstms import rnn_layer, stacked_rnn, stacked_lnlstm

    models = [
      rnn_layer(input_size, hidden_size).eval(),
      stacked_rnn(input_size, hidden_size, num_layers).eval(),
      stacked_lnlstm(input_size, hidden_size, num_layers).eval()
    ]

    for raw_model in models:
        script_module = torch.jit.script(raw_model)
        mod, params = from_pytorch(script_module, input_shapes, input_types)

        # comp = relay.backend.vm.VMCompiler()
        # opt_mod, _ = comp.optimize(mod, "llvm", params)
        # print(opt_mod["main"])
        # continue

        for i in range(5):
            inp = torch.randn(seq_len, batch, input_size)
            states = [(torch.randn(batch, hidden_size),
                       torch.randn(batch, hidden_size))
                      for _ in range(num_layers)]

            with torch.no_grad():
                pt_result = raw_model(inp.clone(), states[0])

            params[input_name] = inp.numpy()
            params["states"] = (st.numpy() for st in states[0])

            run_and_compare(mod, params, pt_result)


# doesn't work, need to wait for fixed size tensor list
# custom_lstm_test()
simple_rnn_test()
