import argparse
import onnx
from onnx_opcounter import calculate_params, calculate_flops


def main():
    parser = argparse.ArgumentParser(description='ONNX opcounter')
    parser.add_argument('model', type=str, help='Path to an ONNX model.')
    parser.add_argument('--calculate-flops', action='store_true', help='Calculate FLOPS.')
    args = parser.parse_args()

    model = onnx.load(args.model)

    print('Number of parameters in the model: {}'.format(calculate_params(model)))

    if args.calculate_flops:
        print('Number of FLOPS in the model: {}'.format(calculate_flops(model)))
