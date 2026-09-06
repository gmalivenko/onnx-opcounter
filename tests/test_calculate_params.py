import torch
from torch import nn
import pytest
import tempfile
import os
from onnx_opcounter import calculate_params
import onnx


def export_onnx(model, input):
    tmp = tempfile.TemporaryDirectory()
    path = os.path.join(tmp.name, "_model.onnx")
    torch.onnx.export(model, input, path, input_names=['input'],
                      output_names=['output'], opset_version=16, dynamo=False)
    try:
        return onnx.load_model(path)
    finally:
        tmp.cleanup()


def check_params(model, input, expected_params):
    onnx_params = calculate_params(export_onnx(model, input))
    assert int(onnx_params) == int(expected_params)


def torch_params(model):
    return sum(p.numel() for p in model.parameters())


@pytest.mark.parametrize('kernel_size', [1, 3, 5, 7])
@pytest.mark.parametrize('padding', [0, 1, 3, 5])
@pytest.mark.parametrize('stride', [1])
@pytest.mark.parametrize('bias', [True, False])
@pytest.mark.parametrize('dilation', [1, 2, 3])
@pytest.mark.parametrize('groups', [1, 2, 3])
def test_conv2d_case1(kernel_size, padding, stride, bias, dilation, groups):
    model = nn.Sequential(nn.Conv2d(
        groups * 3, groups, kernel_size=kernel_size, padding=padding,
        stride=stride, bias=bias, dilation=dilation, groups=groups
    ),)
    model.eval()

    input = torch.randn((1, groups * 3, 224, 224))
    check_params(model, input, torch_params(model))


@pytest.mark.parametrize('kernel_size', [1, 3, 5, 7])
@pytest.mark.parametrize('padding', [0, 1, 3, 5])
@pytest.mark.parametrize('stride', [1])
@pytest.mark.parametrize('bias', [True, False])
@pytest.mark.parametrize('dilation', [1, 2, 3])
@pytest.mark.parametrize('groups', [1, 2, 3])
def test_convtranspose2d_case1(kernel_size, padding, stride, bias, dilation, groups):
    model = nn.Sequential(nn.ConvTranspose2d(
        groups * 3, groups, kernel_size=kernel_size, padding=padding,
        stride=stride, bias=bias, dilation=dilation, groups=groups
    ),)
    model.eval()

    input = torch.randn((1, groups * 3, 224, 224))
    check_params(model, input, torch_params(model))


@pytest.mark.parametrize('inputs', [1, 32, 64, 128, 256])
@pytest.mark.parametrize('outputs', [1, 32, 64, 128])
@pytest.mark.parametrize('bias', [True, False])
def test_linear_case1(inputs, outputs, bias):
    model = nn.Sequential(nn.Linear(
        inputs, outputs, bias=bias
    ),)
    model.eval()

    input = torch.randn((1, inputs))
    check_params(model, input, torch_params(model))


@pytest.mark.parametrize('planes', [1, 32, 64, 128, 256])
@pytest.mark.parametrize('size', [1, 32, 64, 128, 256])
@pytest.mark.parametrize('affine', [True, False])
def test_bn_case1(planes, size, affine):
    model = nn.Sequential(nn.BatchNorm1d(
        planes, affine=affine
    ),)
    model.eval()

    input = torch.randn((1, planes, size))
    check_params(model, input, 2 * planes)
