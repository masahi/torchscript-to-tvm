import itertools
import numpy as np
import torch
import tvm
from tvm.relay import expr as _expr
from tvm.relay import analysis as _analysis
from tvm.relay import module as _module
from tvm.relay.loops import while_loop
from tvm.relay import op as _op

from relay_op_conversion import convert_map, wrap_const


def parse_inputs(graph_inputs, input_shapes):
    ir_inputs = list(graph_inputs)
    ir_names = [i.debugName() for i in ir_inputs]
    input_vars = {}

    for input_name, ir_input in zip(input_shapes, ir_inputs[1:]):
        input_shape = input_shapes[input_name]
        ir_input.setDebugName(input_name)
        input_vars[input_name] = _expr.var(input_name,
                                           shape=input_shapes[input_name])
    # Add self (first input of a PyTorch graph) to inputs
    input_shape = [3]
    tensor = tvm.nd.array(np.zeros(input_shape).astype(np.float32))
    input_name = ir_names[0]  # self.1
    input_vars[input_name] = tensor

    return input_vars


def get_tensor_and_var(torch_tensor, name):
    tensor = tvm.nd.array(torch_tensor.cpu().numpy())
    var = _expr.var(name, shape=tensor.shape)
    return tensor, var


def get_output_name(node):
    assert node.outputsSize() == 1
    return node.output().debugName()


def get_output_names(node):
    return [output.debugName() for output in node.outputs()]


def get_input_names(node):
    return [inp.debugName() for inp in node.inputs()]


def getattr_attr_name(node):
    attribute_names = node.attributeNames()
    assert(len(attribute_names) == 1)
    attr_name = node.s(attribute_names[0])
    return attr_name


def get_attr_chains(root_getattr_node):
    """Returns chains of attribute access starting from root_getattr_node

    For example, given attribute "block", as in "self.block" when "self" points
    to the top level torch.nn.Module, it returns lists of attribute "chains",
    e.g. ['block', '2'], ['block', '1'], ['block', '0', '_packed_params']

    These sets of attributes form full attribute accessors. For example,
    "self.block.1", "self.block.2" will return the second and third submodule,
    and "self.block.0._packed_params" will return the parameters of the first
    submodule.
    """
    def concat_lists(lists):
        return itertools.chain.from_iterable(lists)

    def inner(current, accum):
        users = [use.user for use in current.output().uses()]
        next_attrs = [user for user in users if user.kind() == "prim::GetAttr"]

        if not users or not next_attrs:
            # no next GetAttr -> this is the last attr
            return [accum]

        return concat_lists([inner(nxt, accum + [nxt]) for nxt in next_attrs])

    return inner(root_getattr_node, [root_getattr_node])


def get_full_attr_name(getattrs):
    return ".".join([getattr_attr_name(node) for node in getattrs])


def parse_params(graph, state_dict):
    getattr_nodes = graph.findAllNodes("prim::GetAttr", recurse=True)
    params = {}
    param_tensors = {}
    seen = set()

    for node in getattr_nodes:
        if get_output_name(node) in seen:
            continue

        for getattrs in get_attr_chains(node):
            seen.update(map(get_output_name, getattrs))

            full_attr = get_full_attr_name(getattrs)
            full_attr_node_name = get_output_name(getattrs[-1])

            if full_attr in state_dict:
                torch_tensor = state_dict[full_attr]
                tensor, var = get_tensor_and_var(torch_tensor,
                                                 full_attr_node_name)
                param_tensors[full_attr_node_name] = tensor
                params[full_attr_node_name] = var

    return params, param_tensors


def get_input_types(op_node):
    input_list_types = []
    for input_node in op_node.inputs():
        in_ty = input_node.type()
        input_node_kind = in_ty.kind()
        if input_node_kind == 'TensorType':
            if in_ty.scalarType() is None:
                input_list_types.append('float')
            else:
                input_list_types.append(in_ty.scalarType().lower())
        elif input_node_kind == 'ListType':
            input_list_types.append(str(in_ty.getElementType()).lower())
        elif input_node_kind in ['IntType', 'FloatType', 'BoolType',
                                 'StringType', 'OptionalType']:
            input_list_types.append(str(in_ty).lower())
        else:
            input_list_types.append('UnsupportedType')

    if op_node.kind() in ['aten::ones', 'aten::zeros']:
        node_type = op_node.output().type()
        input_list_types[0] = node_type.scalarType().lower()

    return input_list_types


def get_constant(node):
    attribute_names = node.attributeNames()
    num_attributes = len(attribute_names)

    if num_attributes == 1:
        attr_name = attribute_names[0]
        ty = node.output().type().kind()

        if ty == "IntType" or ty == "BoolType":
            return node.i(attr_name)
        elif ty == "FloatType":
            return node.f(attr_name)
        elif ty == "TensorType":
            tensor = node.t(attr_name)
            if len(tensor.shape) == 0:  # tensor(0.1)
                return float(tensor)
            return tensor
        elif ty == "DeviceObjType":
            return node.s(attr_name)
        elif ty == "FunctionType":
            return None
        else:
            print(ty, node)
            assert False  # TODO: handle other types
    else:
        assert num_attributes == 0
        return None


def parse_ops(nodes):
    ops = {}
    op_inputs_types = {}
    consts = {}
    # Traverse nodes and add to graph
    for node in nodes:
        if node.outputsSize() > 1:
            node_name = "_".join(get_output_names(node))
        else:
            node_name = get_output_name(node)
            if node.kind() == "prim::Constant":
                consts[node_name] = get_constant(node)

        if node.kind() != "prim::GetAttr":
            ops[node_name] = node
            op_inputs_types[node_name] = get_input_types(node)

    return consts, ops, op_inputs_types


def get_input_node_names(op_node, output_index_map):
    return [output_index_map[name] for name in get_input_names(op_node)]


def get_op_inputs(op_node, outputs, output_index_map):
    input_names = get_input_node_names(op_node, output_index_map)
    return [outputs[name] for name in input_names]


def is_int_list(lst):
    return all([isinstance(i, int) for i in lst])


def run_jit_passes(graph):
    torch._C._jit_pass_inline(graph)


def update_outputs_from_pairs(name_output_pairs, outputs, output_index_map):
    for output_name, output in name_output_pairs:
        output_index_map[output_name] = len(outputs)
        outputs.append(output)


def get_free_vars_from_block(block):
    block_inp_names = get_input_names(block)
    bound_names = block_inp_names
    free_vars = set()

    for node in block.nodes():
        inp_names = get_input_names(node)
        list_diff = [name for name in inp_names if name not in bound_names]
        free_vars.update(list_diff)
        bound_names += get_output_names(node)

    return list(free_vars)


def parse_block(block, consts, op_in_types, outputs, output_index_map):
    consts_block, ops, op_in_types_block = parse_ops(block.nodes())
    consts_block.update(consts)
    op_in_types_block.update(op_in_types)
    return parse_operators(ops, consts_block, op_in_types_block,
                           outputs, output_index_map)


def parse_loop(op_node, consts, op_in_types, outputs, output_index_map):

    def get_input(index):
        var_name = op_node.inputsAt(index).debugName()
        if var_name in consts:
            return _expr.const(consts[var_name])
        assert var_name in output_index_map
        output_ind = output_index_map[var_name]
        out = outputs[output_ind]
        if isinstance(out, _expr.Expr):  # TODO: remove this condition
            return out
        return _expr.const(out)

    max_loop_count = get_input(0)
    init_cond = get_input(1)
    num_loop_var = len(list(op_node.inputs())) - 2
    init_vals = [get_input(i + 2) for i in range(num_loop_var)]

    body_block = list(op_node.blocks())[0]
    inames = get_input_names(body_block)
    loop_input_vals = [_expr.const(1), init_cond] + init_vals
    update_outputs_from_pairs(zip(inames, loop_input_vals),
                              outputs, output_index_map)

    def cond(*current_vals):
        i = current_vals[0]
        lhs = _op.greater_equal(i, _expr.const(1, 'int32'))
        rhs = _op.less_equal(i, max_loop_count)
        return _op.logical_and(lhs, rhs)

    def body(*current_vals):
        for (i, iname) in enumerate(inames):
            outputs[output_index_map[iname]] = current_vals[i]
        ret = parse_block(body_block, consts, op_in_types,
                          outputs, output_index_map)
        incr = _expr.const(1, 'int32')
        return current_vals[0] + incr, ret

    # free_vars = get_free_vars_from_block(body_block)
    # fixed_vals = [outputs[output_index_map[name]] for name in free_vars]
    # init_vals += [wrap_const(val) for val in fixed_vals]

    loop_count_var = _expr.var(inames[0], shape=(), dtype='int32')
    loop_vars = [_expr.var(name) for name in inames[1:]]  # + free_vars]
    loop = while_loop(cond, [loop_count_var] + loop_vars, body)
    loop_val = loop(_expr.const(1), *init_vals)
    return _expr.TupleGetItem(loop_val, 1)


def parse_operators(operators, consts, op_in_types, outputs, output_index_map):
    for node_name, op_node in operators.items():
        operator = op_node.kind()
        output_index_map[node_name] = len(outputs)
        inputs = get_op_inputs(op_node, outputs, output_index_map)

        if operator == "prim::Constant":
            outputs.append(consts[node_name])
        elif operator == 'prim::ListConstruct' and is_int_list(inputs):
            outputs.append(_expr.var(node_name, shape=inputs))
        elif operator == 'prim::ListConstruct':
            outputs.append(inputs)
        elif operator == "prim::ListUnpack":
            update_outputs_from_pairs(zip(get_output_names(op_node), inputs[0]),
                                      outputs, output_index_map)
        elif operator == "prim::If":
            cond = outputs[output_index_map[op_node.inputsAt(0).debugName()]]
            blocks = list(op_node.blocks())
            true_branch = parse_block(blocks[0], consts, op_in_types,
                                      outputs, output_index_map)
            false_branch = parse_block(blocks[1], consts, op_in_types,
                                       outputs, output_index_map)
            outputs.append(_expr.If(cond, true_branch, false_branch))
        elif operator == "prim::Loop":
            loop = parse_loop(op_node, consts, op_in_types,
                              outputs, output_index_map)
            outputs.append(loop)
        else:
            relay_op = convert_map[operator]
            outputs.append(relay_op(inputs, op_in_types[node_name]))

    return outputs[-1]


def parse_script_module(script_module, input_shapes):
    graph = script_module.graph.copy()
    run_jit_passes(graph)

    params = script_module.state_dict()
    input_vars = parse_inputs(graph.inputs(), input_shapes)
    param_vars, tensors = parse_params(graph, params)
    consts, ops, op_in_types = parse_ops(graph.nodes())

    input_vars.update(param_vars)
    outputs = list(input_vars.values())
    output_index_map = dict(zip(input_vars.keys(), range(len(outputs))))

    body = parse_operators(ops, consts, op_in_types, outputs, output_index_map)
    func = tvm.relay.Function(_analysis.free_vars(body), body)
    tvm_params = {k: tvm.nd.array(v) for k, v in tensors.items()}

    return _module.Module.from_expr(func), tvm_params
