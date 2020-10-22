# import the modules used here in this recipe
import torch
import torch.quantization
import torch.nn as nn
import copy
import os
import time

# define a very, very simple LSTM for demonstration purposes
# in this case, we are wrapping nn.LSTM, one layer, no pre or post processing
# inspired by
# https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html, by Robert Guthrie
# and https://pytorch.org/tutorials/advanced/dynamic_quantization_tutorial.html
class lstm_for_demonstration(nn.Module):
  """Elementary Long Short Term Memory style model which simply wraps nn.LSTM
     Not to be used for anything other than demonstration.
  """
  def __init__(self,in_dim,out_dim,depth):
     super(lstm_for_demonstration,self).__init__()
     self.lstm = nn.LSTM(in_dim,out_dim,depth)

  def forward(self,inputs,hidden):
     out,hidden = self.lstm(inputs,hidden)
     return out, hidden


torch.manual_seed(29592)  # set the seed for reproducibility

#shape parameters
model_dimension=8
sequence_length=20
batch_size=1
lstm_depth=1

# random data for input
inputs = torch.randn(sequence_length,batch_size,model_dimension)
# hidden is actually is a tuple of the initial hidden state and the initial cell state
hidden = (torch.randn(lstm_depth,batch_size,model_dimension), torch.randn(lstm_depth,batch_size,model_dimension))

 # here is our floating point instance
float_lstm = lstm_for_demonstration(model_dimension, model_dimension,lstm_depth)

# this is the call that does the work
quantized_lstm = torch.quantization.quantize_dynamic(
    float_lstm, {nn.LSTM, nn.Linear}, dtype=torch.qint8
)

# show the changes that were made
print('Here is the floating point version of this module:')
print(float_lstm)
print('')
print('and now the quantized version:')
print(quantized_lstm)

# compare the performance
# print("Floating point FP32")
# %timeit float_lstm.forward(inputs, hidden)

# print("Quantized INT8")
# %timeit quantized_lstm.forward(inputs,hidden)

script_module = torch.jit.trace(quantized_lstm, (inputs, hidden))

from tvm import relay
input_shapes = [("inputs", inputs.shape), ("hidden0", hidden[0].shape), ("hidden1", hidden[0].shape)]
relay.frontend.from_pytorch(script_module, input_shapes)
