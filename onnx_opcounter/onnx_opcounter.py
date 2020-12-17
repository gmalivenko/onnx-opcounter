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
        except:
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


# def calculate_flops(model: onnx.ModelProto) -> int:
#     onnx_nodes = model.graph.node
#
#     onnx_weights = model.graph.initializer
#     node_names = [i.name for i in onnx_nodes]
#     node_types = [i.op_type for i in onnx_nodes]
#
#     weights = {}
#     for onnx_w in onnx_weights:
#         try:
#             if len(onnx_w.ListFields()) < 4:
#                 onnx_extracted_weights_name = onnx_w.ListFields()[1][1]
#             else:
#                 onnx_extracted_weights_name = onnx_w.ListFields()[2][1]
#             weights[onnx_extracted_weights_name] = weight = numpy_helper.to_array(onnx_w)
#         except:
#             onnx_extracted_weights_name = onnx_w.ListFields()[3][1]
#             weights[onnx_extracted_weights_name] = weight = numpy_helper.to_array(onnx_w)
#         print(onnx_extracted_weights_name, weight.shape)
#
#     for node in onnx_nodes:
#       print(onnx_node_attributes_to_dict(node.attribute))
#
#     for node in node_names:
#         intermediate_tensor_name = node
#         intermediate_layer_value_info = onnx.helper.ValueInfoProto()
#         intermediate_layer_value_info.name = intermediate_tensor_name
#         model.graph.output.extend([intermediate_layer_value_info])
#
#     onnx.save(model, '+all-intermediate.onnx')
#
#     print(node_names, node_types)
#     print(onnx_nodes[1].input)
#     # intermediate_tensor_name = "convolution_output74"
#     # intermediate_layer_value_info = helper.ValueInfoProto()
#     # intermediate_layer_value_info.name = intermediate_tensor_name
#     # model.graph.output.extend([intermediate_layer_value_info])
#     # onnx.save(model, model_path)
#
#     # # Load model itself
#     # model = onnx.load(model_path)
#
#     print(len(node_names))
#     sess = rt.InferenceSession('+all-intermediate.onnx')
#     output = sess.run(node_names, {'data': np.zeros((1,3,224,224), dtype=np.float32)})
#     print(len(output))