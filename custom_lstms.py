# originally from https://github.com/pytorch/pytorch/blob/master/benchmarks/fastrnns/custom_lstms.py

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.jit as jit
from typing import List, Tuple
from torch import Tensor


class LayerNormLSTMCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))

        ln = nn.LayerNorm

        self.layernorm_i = ln(4 * hidden_size)
        self.layernorm_h = ln(4 * hidden_size)
        self.layernorm_c = ln(hidden_size)

    @jit.script_method
    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        hx, cx = state
        igates = self.layernorm_i(torch.mm(input, self.weight_ih.t()))
        hgates = self.layernorm_h(torch.mm(hx, self.weight_hh.t()))
        gates = igates + hgates
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = self.layernorm_c((forgetgate * cx) + (ingate * cellgate))
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)


class LSTMLayer(jit.ScriptModule):
    def __init__(self, cell, *cell_args):
        super().__init__()
        self.cell = cell(*cell_args)

    @jit.script_method
    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        outputs = []
        for i in range(input.size(0)):
            out, state = self.cell(input[i], state)
            outputs += [out]
        return torch.stack(outputs), state


class ReverseLSTMLayer(jit.ScriptModule):
    def __init__(self, cell, *cell_args):
        super(ReverseLSTMLayer, self).__init__()
        self.cell = cell(*cell_args)

    @jit.script_method
    def forward(self, inputs, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        outputs = jit.annotate(List[Tensor], [])
        seq_len = inputs.size(0)
        for i in range(seq_len):
            out, state = self.cell(inputs[seq_len - i - 1], state)
            # workaround for the lack of list rev support
            outputs = [out] + outputs
        return torch.stack(outputs), state


class BidirLSTMLayer(jit.ScriptModule):
    __constants__ = ['directions']

    def __init__(self, cell, *cell_args):
        super(BidirLSTMLayer, self).__init__()
        self.directions = nn.ModuleList([
            LSTMLayer(cell, *cell_args),
            ReverseLSTMLayer(cell, *cell_args),
        ])

    @jit.script_method
    def forward(self, input, states):
        # type: (Tensor, List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]
        # List[LSTMState]: [forward LSTMState, backward LSTMState]
        outputs = jit.annotate(List[Tensor], [])
        output_states = jit.annotate(List[Tuple[Tensor, Tensor]], [])
        for (i, direction) in enumerate(self.directions):
            state = states[i]
            out, out_state = direction(input, state)
            outputs += [out]
            output_states += [out_state]
        # tensor array concat assumes axis == 0 for now
        # return torch.cat(outputs, -1), output_states
        return torch.cat(outputs, 0), output_states


def init_stacked_lstm(num_layers, layer, first_layer_args, other_layer_args):
    layers = [layer(*first_layer_args)] + [layer(*other_layer_args)
                                           for _ in range(num_layers - 1)]
    return nn.ModuleList(layers)


class StackedLSTM(jit.ScriptModule):
    __constants__ = ['layers']  # Necessary for iterating through self.layers

    def __init__(self, num_layers, layer, first_layer_args, other_layer_args):
        super().__init__()
        self.layers = init_stacked_lstm(num_layers, layer, first_layer_args,
                                        other_layer_args)

    @jit.script_method
    def forward(self, input, states):
        # type: (Tensor, List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]
        # List[LSTMState]: One state per layer
        output_states = jit.annotate(List[Tuple[Tensor, Tensor]], [])
        output = input
        for (i, rnn_layer) in enumerate(self.layers):
            state = states[i]
            output, out_state = rnn_layer(output, state)
            output_states += [out_state]
        return output, output_states


class StackedBidirLSTM(jit.ScriptModule):
    __constants__ = ['layers']  # Necessary for iterating through self.layers

    def __init__(self, num_layers, layer, first_layer_args, other_layer_args):
        super(StackedBidirLSTM, self).__init__()
        self.layers = init_stacked_lstm(num_layers, layer, first_layer_args,
                                        other_layer_args)

    @jit.script_method
    def forward(self, input, states):
        # type: (Tensor, List[List[Tuple[Tensor, Tensor]]]) -> Tuple[Tensor, List[List[Tuple[Tensor, Tensor]]]]
        # List[List[LSTMState]]: The outer list is for layers,
        #                        inner list is for directions.
        output_states = jit.annotate(List[List[Tuple[Tensor, Tensor]]], [])
        output = input
        for (i, rnn_layer) in enumerate(self.layers):
            state = states[i]
            output, out_state = rnn_layer(output, state)
            output_states += [out_state]
        return output, output_states


def lstm(input_size, hidden_size):
    return LSTMLayer(LayerNormLSTMCell, input_size, hidden_size)


def stacked_lstm(input_size, hidden_size, num_layers):
    return StackedLSTM(num_layers, LSTMLayer,
                       first_layer_args=[LayerNormLSTMCell, input_size, hidden_size],
                       other_layer_args=[LayerNormLSTMCell, hidden_size, hidden_size])


def bidir_lstm(input_size, hidden_size):
    return BidirLSTMLayer(LayerNormLSTMCell, input_size, hidden_size)


def stacked_bidir_lstm(input_size, hidden_size, num_layers):
    return StackedBidirLSTM(num_layers, BidirLSTMLayer,
                            first_layer_args=[LayerNormLSTMCell, input_size, hidden_size],
                            other_layer_args=[LayerNormLSTMCell, hidden_size * 2, hidden_size])
