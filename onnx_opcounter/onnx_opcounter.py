import onnx
import numpy as np
from onnx import numpy_helper
import os
import tempfile
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

    return int(params)


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


FREE_OPS = frozenset({
    'Concat', 'ConcatFromSequence', 'Flatten', 'Pad', 'Expand', 'DepthToSpace',
    'SpaceToDepth', 'GatherElements', 'GatherND', 'ScatterElements', 'Scatter',
    'Compress', 'ReverseSequence', 'Col2Im', 'MaxUnpool', 'Dropout', 'CastLike',
    'Trilu', 'CenterCropPad', 'BitCast', 'TensorScatter', 'ImageDecoder',
    'RegexFullMatch', 'StringConcat', 'StringNormalizer', 'StringSplit',
    'Size', 'ConstantOfShape', 'Range', 'NonZero', 'EyeLike', 'Unique', 'OneHot',
    'NonMaxSuppression',
    'Optional', 'OptionalGetElement', 'OptionalHasElement', 'SequenceAt',
    'SequenceConstruct', 'SequenceEmpty', 'SequenceErase', 'SequenceInsert',
    'SequenceLength', 'SplitToSequence',
    'If', 'Loop', 'Scan', 'SequenceMap',
})

ELEMENTWISE_OPS = frozenset({
    'Sub', 'Mul', 'Div', 'Mod', 'Pow', 'Abs', 'Neg', 'Reciprocal', 'Sqrt', 'Exp',
    'Log', 'Sum', 'Mean', 'Max', 'Min', 'Clip', 'Round', 'Floor', 'Ceil',
    'Equal', 'Greater', 'GreaterOrEqual', 'Less', 'LessOrEqual', 'And', 'Or',
    'Xor', 'Not', 'Where', 'IsInf', 'IsNaN', 'BitShift', 'BitwiseAnd',
    'BitwiseNot', 'BitwiseOr', 'BitwiseXor', 'CumSum', 'CumProd',
    'Sin', 'Cos', 'Tan', 'Asin', 'Acos', 'Atan', 'Sinh', 'Cosh', 'Tanh',
    'Asinh', 'Acosh', 'Atanh',
    'Elu', 'Celu', 'Selu', 'Gelu', 'Mish', 'Swish', 'HardSwish', 'HardSigmoid',
    'LeakyRelu', 'PRelu', 'Sigmoid', 'Softplus', 'Softsign', 'ThresholdedRelu',
    'Shrink', 'Erf', 'Softmax', 'LogSoftmax', 'Hardmax',
    'QuantizeLinear', 'DequantizeLinear', 'DynamicQuantizeLinear',
    'GridSample', 'RoiAlign', 'MaxRoiPool', 'LpNormalization', 'RotaryEmbedding',
    'AffineGrid',
    'Bernoulli', 'Multinomial', 'RandomNormal', 'RandomNormalLike',
    'RandomUniform', 'RandomUniformLike',
    'BlackmanWindow', 'HammingWindow', 'HannWindow', 'MelWeightMatrix',
    'TfIdfVectorizer',
})

REDUCTION_OPS = frozenset({
    'ReduceL1', 'ReduceL2', 'ReduceLogSum', 'ReduceLogSumExp', 'ReduceMax',
    'ReduceMean', 'ReduceMin', 'ReduceProd', 'ReduceSum', 'ReduceSumSquare',
    'ArgMax', 'ArgMin', 'TopK', 'Det',
    'NegativeLogLikelihoodLoss', 'SoftmaxCrossEntropyLoss',
})


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

        try:
            import onnxruntime as rt
        except ImportError as import_error:
            raise ImportError(
                'Shape inference failed, so the model has to be run to discover '
                'intermediate shapes, but onnxruntime is not installed. '
                'Install it with: pip install onnx_opcounter[runtime]'
            ) from import_error

        provider = 'CPUExecutionProvider'

        with tempfile.TemporaryDirectory() as tmp:
            intermediate_model = os.path.join(tmp, 'all-intermediate.onnx')
            onnx.save(model, intermediate_model)

            sess = rt.InferenceSession(intermediate_model, providers=[provider])
            start = time.time()
            output = sess.run(list(graph_outputs.keys()), input_sample)
            print("inference(s):", time.time() - start)

        output_shapes = {**{k: input_sample[k].shape for k in input_sample},
                        **{i: output[output_mapping[i]].shape for i in output_mapping}
                        }

    for w in model.graph.initializer:
        output_shapes[w.name] = list(w.dims)

    def conv_macs(node, input_shape, output_shape, attrs):
        if 'kernel_shape' in attrs:
            kernel_ops = np.prod(attrs['kernel_shape'])
        else:
            weight = 3 if node.op_type == 'QLinearConv' else 1
            kernel_ops = np.prod(output_shapes[node.input[weight]][2:])

        group = 1
        if 'group' in attrs:
            group = attrs['group']

        in_channels = input_shape[1]

        return np.prod(output_shape) * (in_channels // group * kernel_ops)

    def gemm_macs(node, input_shape, output_shape, attrs):
        return np.prod(input_shape) * output_shape[-1]

    def pool_macs(node, input_shape, output_shape, attrs):
        kernel_ops = np.prod(attrs['kernel_shape']) if 'kernel_shape' in attrs else 1
        return np.prod(output_shape) * kernel_ops

    def global_pool_macs(node, input_shape, output_shape, attrs):
        return np.prod(input_shape)

    def rnn_macs(node, input_shape, output_shape, attrs):
        layout = attrs.get('layout', 0)
        seq_length = input_shape[1] if layout else input_shape[0]
        batch_size = input_shape[0] if layout else input_shape[1]

        gate_weights = np.prod(output_shapes[node.input[1]])
        recurrent_weights = np.prod(output_shapes[node.input[2]])
        return seq_length * batch_size * (gate_weights + recurrent_weights)

    def einsum_macs(node, input_shape, output_shape, attrs):
        equation = attrs['equation']
        if isinstance(equation, bytes):
            equation = equation.decode('utf-8')
        equation = equation.replace(' ', '')

        if '->' not in equation or '...' in equation:
            return np.prod(output_shape)

        terms, result = equation.split('->')
        sizes = {}
        for term, name in zip(terms.split(','), node.input):
            for label, dim in zip(term, output_shapes[name]):
                sizes[label] = dim

        contracted = 1
        for label, dim in sizes.items():
            if label not in result:
                contracted *= dim

        return np.prod(output_shape) * contracted

    def attention_macs(node, input_shape, output_shape, attrs):
        query, key, value = (output_shapes[i] for i in node.input[:3])
        if not (len(query) == len(key) == len(value) == 4):
            return np.prod(output_shape)

        batch, heads, q_length, head_size = query
        kv_length = key[2]
        scores = batch * heads * q_length * kv_length * head_size
        context = batch * heads * q_length * kv_length * value[3]
        return scores + context

    def bn_macs(node, input_shape, output_shape, attrs):
        batch_macs = np.prod(output_shape)
        if len(node.input) == 5:
            batch_macs *= 2
        return batch_macs

    def norm_macs(node, input_shape, output_shape, attrs):
        passes = 3 if node.op_type == 'RMSNormalization' else 5
        affine = len(node.input) - 1
        return np.prod(output_shape) * (passes + affine)

    def lrn_macs(node, input_shape, output_shape, attrs):
        return np.prod(output_shape) * attrs.get('size', 1)

    def upsample_macs(node, input_shape, output_shape, attrs):
        mode = attrs.get('mode', b'nearest')
        if isinstance(mode, bytes):
            mode = mode.decode('utf-8')
        if mode == 'linear':
            return np.prod(output_shape) * 11
        return 0

    def dft_macs(node, input_shape, output_shape, attrs):
        length = max(output_shape[-2], 2) if len(output_shape) >= 2 else 2
        return np.prod(output_shape) * int(np.log2(length))

    def elementwise_macs(node, input_shape, output_shape, attrs):
        return np.prod(output_shape)

    def reduce_macs(node, input_shape, output_shape, attrs):
        return np.prod(input_shape)

    def no_macs(*args, **kwargs):
        return 0

    mac_calculators = {op: no_macs for op in FREE_OPS}
    mac_calculators.update({op: elementwise_macs for op in ELEMENTWISE_OPS})
    mac_calculators.update({op: reduce_macs for op in REDUCTION_OPS})
    mac_calculators.update({
        'Conv': conv_macs,
        'ConvTranspose': conv_macs,
        'DeformConv': conv_macs,
        'QLinearConv': conv_macs,
        'ConvInteger': conv_macs,
        'Constant': no_macs,
        'Identity': no_macs,
        'Gemm': gemm_macs,
        'MatMul': gemm_macs,
        'QLinearMatMul': gemm_macs,
        'MatMulInteger': gemm_macs,
        'Einsum': einsum_macs,
        'Attention': attention_macs,
        'LinearAttention': attention_macs,
        'CausalConvWithState': conv_macs,
        'DFT': dft_macs,
        'STFT': dft_macs,
        'MaxPool': pool_macs,
        'AveragePool': pool_macs,
        'LpPool': pool_macs,
        'GlobalAveragePool': global_pool_macs,
        'GlobalMaxPool': global_pool_macs,
        'GlobalLpPool': global_pool_macs,
        'LSTM': rnn_macs,
        'GRU': rnn_macs,
        'RNN': rnn_macs,
        'BatchNormalization': bn_macs,
        'LayerNormalization': norm_macs,
        'GroupNormalization': norm_macs,
        'InstanceNormalization': norm_macs,
        'RMSNormalization': norm_macs,
        'MeanVarianceNormalization': norm_macs,
        'LRN': lrn_macs,
        'Relu': elementwise_macs,
        'Add': elementwise_macs,
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
    })

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

    return int(macs)
