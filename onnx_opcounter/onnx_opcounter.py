import onnx
import onnxruntime as rt
import numpy as np
from onnx import numpy_helper


def calculate_params(model: onnx.ModelProto) -> int:
    onnx_weights = model.graph.initializer
    params = 0

    for onnx_w in onnx_weights:
        try:
            weight = numpy_helper.to_array(onnx_w)
            params += np.prod(weight.shape)
        except Exception as _:
            pass

    return params


def onnx_node_attributes_to_dict(args):
    """
    Parse ONNX attributes to Python dictionary
    :param args: ONNX attributes object
    :return: Python dictionary
    """
    def onnx_attribute_to_dict(onnx_attr):
        """
        Parse ONNX attribute
        :param onnx_attr: ONNX attribute
        :return: Python data type
        """
        if onnx_attr.HasField('t'):
            return numpy_helper.to_array(getattr(onnx_attr, 't'))

        for attr_type in ['f', 'i', 's']:
            if onnx_attr.HasField(attr_type):
                return getattr(onnx_attr, attr_type)

        for attr_type in ['floats', 'ints', 'strings']:
            if getattr(onnx_attr, attr_type):
                return list(getattr(onnx_attr, attr_type))
    return {arg.name: onnx_attribute_to_dict(arg) for arg in args}


def calculate_macs(model: onnx.ModelProto) -> int:
    onnx_nodes = model.graph.node
    onnx_weights = model.graph.initializer

    graph_weights = [w.name for w in onnx_weights]
    graph_inputs = [i.name for i in model.graph.input]
    graph_outputs = [i.name for i in model.graph.output]

    input_sample = {}
    type_mapping = {
        1: np.float32,
        7: np.int64,
    }

    for graph_input in model.graph.input:
        if graph_input.name not in graph_weights:
            input_sample[graph_input.name] = \
                np.zeros([i.dim_value for i in graph_input.type.tensor_type.shape.dim],
                         dtype=type_mapping[graph_input.type.tensor_type.elem_type])

    def get_mapping_for_node(node, graph_outputs):
        for output in node.output:
            if output in graph_outputs:
                return output
        return node.name

    output_name_mapping = {node.name: get_mapping_for_node(node, graph_outputs) for node in onnx_nodes}
    output_mapping = {}

    for name in output_name_mapping:
        output = output_name_mapping[name]
        if output in graph_outputs:
            output_mapping[name] = graph_outputs.index(output)
        else:
            intermediate_layer_value_info = onnx.helper.ValueInfoProto()
            intermediate_layer_value_info.name = output
            model.graph.output.extend([intermediate_layer_value_info])
            graph_outputs.append(output)
            output_mapping[name] = graph_outputs.index(output)

        print(name, '->', output, 'index', output_mapping[name])

    onnx.save(model, '+all-intermediate.onnx')

    sess = rt.InferenceSession('+all-intermediate.onnx')
    output = sess.run(graph_outputs, input_sample)

    output_shapes = {**{k: input_sample[k].shape for k in input_sample},
                     **{i: output[output_mapping[i]].shape for i in output_mapping}
                     }

    def conv_macs(node, input_shape, output_shape, attrs):
        kernel_ops = np.prod(attrs['kernel_shape'])  # Kw x Kh
        bias_ops = len(node.input) == 3

        group = 1
        if 'group' in attrs:
            group = attrs['group']

        in_channels = input_shape[1]

        return np.prod(output_shape) * (in_channels // group * kernel_ops + bias_ops)

    def gemm_macs(node, input_shape, output_shape, attrs):
        return np.prod(input_shape) * np.prod(output_shape)

    def bn_macs(node, input_shape, output_shape, attrs):
        batch_macs = np.prod(output_shape)
        if len(node.input) == 5:
            batch_macs *= 2
        return batch_macs

    def upsample_macs(node, input_shape, output_shape, attrs):
        if 'mode' in attrs:
            if attrs['mode'].decode('utf-8') == 'nearest':
                return 0
            if attrs['mode'].decode('utf-8') == 'linear':
                return np.prod(output_shape) * 11
        else:
            return 0

    def relu_macs(node, input_shape, output_shape, attrs):
        return np.prod(input_shape)

    def no_macs(*args, **kwargs):
        return 0

    mac_calculators = {
        'Conv': conv_macs,
        'Gemm': gemm_macs,
        'MatMul': gemm_macs,
        'BatchNormalization': bn_macs,
        'Relu': relu_macs,
        'Add': relu_macs,
        'Reshape': no_macs,
        'Upsample': upsample_macs,
    }

    macs = 0
    for node in onnx_nodes:
        node_output_shape = output_shapes[node.name]
        node_input_shape = output_shapes[node.input[0]]
        macs += mac_calculators[node.op_type](
            node, node_input_shape, node_output_shape, onnx_node_attributes_to_dict(node.attribute)
        )
    return macs
