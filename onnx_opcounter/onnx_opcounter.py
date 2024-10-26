import onnx
import onnxruntime as rt
import numpy as np
from onnx import numpy_helper
import time


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
    orig_model = model
    model = onnx.ModelProto()
    model.CopyFrom(orig_model)

    onnx_nodes = model.graph.node
    onnx_weights = model.graph.initializer

    graph_weights = [w.name for w in onnx_weights]
    graph_outputs = {i.name: i.name for i in model.graph.output}

    input_sample = {}
    type_mapping = {
        1: np.float32,
        7: np.int64,
        11: np.float64
    }

    def to_dims(v: onnx.ValueInfoProto) -> [int]:
        return [i.dim_value for i in v.type.tensor_type.shape.dim]

    for graph_input in model.graph.input:
        if graph_input.name not in graph_weights:
            input_sample[graph_input.name] = \
                np.zeros(to_dims(graph_input),
                         dtype=type_mapping[graph_input.type.tensor_type.elem_type])

    output_mapping = {k: i for i, k in enumerate(graph_outputs)}
    for n in onnx_nodes:
        for o in n.output:
            if o in graph_outputs:
                continue
            intermediate_layer_value_info = onnx.helper.ValueInfoProto()
            intermediate_layer_value_info.name = o
            model.graph.output.extend([intermediate_layer_value_info])
            output_mapping[o] = len(graph_outputs)
            graph_outputs[o] = o
            assert len(model.graph.output) == len(graph_outputs)
            assert len(model.graph.output) == len(output_mapping)


    try:
        shaped_model = onnx.ModelProto()
        shaped_model.CopyFrom(orig_model)
        del shaped_model.graph.value_info[:]
        shaped_model = onnx.shape_inference.infer_shapes(shaped_model, data_prop=True, strict_mode=True)

        output_shapes = {**{i.name: to_dims(i) for i in shaped_model.graph.value_info},
                         **{i.name: to_dims(i) for i in shaped_model.graph.input},
                         **{i.name: to_dims(i) for i in shaped_model.graph.output},
                         }
    except onnx.shape_inference.InferenceError as e:
        print("Shape inference failure:", e)

        onnx.save(model, '+all-intermediate.onnx')

        provider = 'CPUExecutionProvider'
        # if 'CUDAExecutionProvider' in rt.get_available_providers():
        #     provider = 'CUDAExecutionProvider'

        sess = rt.InferenceSession('+all-intermediate.onnx', providers=[provider])
        start = time.time()
        output = sess.run(list(graph_outputs.keys()), input_sample)
        print("inference(s):", time.time() - start)

        output_shapes = {**{k: input_sample[k].shape for k in input_sample},
                        **{i: output[output_mapping[i]].shape for i in output_mapping}
                        }

    for w in model.graph.initializer:
        output_shapes[w.name] = list(w.dims)

    def conv_macs(node, input_shape, output_shape, attrs):
        kernel_ops = np.prod(attrs['kernel_shape'])  # Kw x Kh
        bias_ops = len(node.input) == 3

        group = 1
        if 'group' in attrs:
            group = attrs['group']

        in_channels = input_shape[1]

        return np.prod(output_shape) * (in_channels // group * kernel_ops)  # + bias_ops

    def gemm_macs(node, input_shape, output_shape, attrs):
        return np.prod(input_shape) * output_shape[-1]

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
        'ConvTranspose': conv_macs,
        'Constant': no_macs,
        'Gemm': gemm_macs,
        'MatMul': gemm_macs,
        'BatchNormalization': bn_macs,
        'Relu': relu_macs,
        'Add': relu_macs,
        'Reshape': no_macs,
        'Slice': no_macs,
        'Shape': no_macs,
        'Gather': no_macs,
        'ScatterND': no_macs,
        'Tile': no_macs,
        'Transpose': no_macs,
        'Sign': no_macs,
        'Squeeze': no_macs,
        'Unsqueeze': no_macs,
        'Split': no_macs,
        'Cast': no_macs,
        'Upsample': upsample_macs,
        'Resize': upsample_macs,
    }

    macs = 0
    unsupported_ops = set()
    for node in onnx_nodes:
        node_output_shape = output_shapes[node.output[0]]
        if node.op_type in mac_calculators:
            node_input_shape = None
            if len(node.input) > 0:
                node_input_shape = output_shapes[node.input[0]]
            macs += mac_calculators[node.op_type](
                node, node_input_shape, node_output_shape, onnx_node_attributes_to_dict(node.attribute)
            )
        else:
            macs += np.prod(node_output_shape)
            if node.op_type in unsupported_ops:
                continue
            print("Unsupported op:", node.op_type)
            unsupported_ops.add(node.op_type)

    return macs
