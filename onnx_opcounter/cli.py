import argparse
import math
import onnx
from onnx_opcounter import calculate_params, calculate_macs


def main():
    parser = argparse.ArgumentParser(description='ONNX opcounter')
    parser.add_argument('model', type=str, help='Path to an ONNX model.')
    parser.add_argument('--calculate-macs', action='store_true', help='Calculate MACs.')
    args = parser.parse_args()

    model = onnx.load(args.model)

    print('Number of parameters in the model: {}'.format(calculate_params(model)))

    if args.calculate_macs:
        macs = calculate_macs(model)
        print('Number of MACs in the model: {}'.format(macs), "log10:", math.log10(macs))
