import torch
from torch import nn

from tvm import relay


rnn = nn.LSTM(10, 20, 2)
input = torch.randn(5, 3, 10)
h0 = torch.randn(2, 3, 20)
c0 = torch.randn(2, 3, 20)
output, (hn, cn) = rnn(input, (h0, c0))

script_module = torch.jit.trace(rnn, (input, (h0, c0)))

print(script_module.graph)
# relay.frontend.from_pytorch(script_module,
