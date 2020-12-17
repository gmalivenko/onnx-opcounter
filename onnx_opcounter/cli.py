import argparse
import onnx
from onnx_opcounter import calculate_params


def main():
    parser = argparse.ArgumentParser(description='ONNX opcounter')
    parser.add_argument('model', type=str, help='Path to an ONNX model.')
    args = parser.parse_args()

    model = onnx.load(args.model)
    print(calculate_params(model))
