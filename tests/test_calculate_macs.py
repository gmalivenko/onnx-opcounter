import torch
import pytest
from torch import nn
import tempfile
import os
from onnx_opcounter import calculate_macs
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


def check_macs(model, input, expected_macs):
    onnx_macs = calculate_macs(export_onnx(model, input))
    assert int(onnx_macs) == int(expected_macs)


def output_numel(model, input):
    with torch.no_grad():
        return model(input).numel()


@pytest.mark.parametrize('inputs', [1, 11, 12])
@pytest.mark.parametrize('outputs', [1, 11, 12])
@pytest.mark.parametrize('kernel_size', [1, 3, 5])
@pytest.mark.parametrize('padding', [0, 1, 2])
@pytest.mark.parametrize('stride', [1, 2])
@pytest.mark.parametrize('bias', [True, False])
@pytest.mark.parametrize('dilation', [1, 2, 3])
@pytest.mark.parametrize('groups', [1, 2, 3])
def test_conv2d_case1(inputs, outputs, kernel_size, padding, stride, bias, dilation, groups):
    model = nn.Sequential(nn.Conv2d(
        inputs * groups, outputs * groups, kernel_size=kernel_size, padding=padding,
        stride=stride, bias=bias, dilation=dilation, groups=groups
    ),)
    model.eval()

    input = torch.randn((1, inputs * groups, 224, 224))
    expected = output_numel(model, input) * inputs * kernel_size * kernel_size
    check_macs(model, input, expected)


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
    expected = output_numel(model, input) * 3 * kernel_size * kernel_size
    check_macs(model, input, expected)


@pytest.mark.parametrize('inputs', [1, 32, 64, 128, 256])
@pytest.mark.parametrize('outputs', [1, 32, 64, 128])
@pytest.mark.parametrize('bias', [True, False])
def test_linear_case1(inputs, outputs, bias):
    model = nn.Sequential(nn.Linear(
        inputs, outputs, bias=bias
    ),)
    model.eval()

    input = torch.randn((1, inputs))
    check_macs(model, input, inputs * outputs)


@pytest.mark.parametrize('inputs', [1, 32, 64, 128, 256])
@pytest.mark.parametrize('affine', [True, False])
def test_bn_case1(inputs, affine):
    model = nn.Sequential(nn.BatchNorm2d(
        inputs, affine=affine
    ))
    model.eval()

    input = torch.randn((1, inputs, 224, 224))
    check_macs(model, input, 2 * output_numel(model, input))


@pytest.mark.parametrize('inputs', [1, 32, 64, 128, 256])
@pytest.mark.parametrize('scale_factor', [1, 2, 3, 5])
@pytest.mark.parametrize('mode', ['bilinear', 'nearest'])
def test_upsample_case1(inputs, scale_factor, mode):
    model = nn.Sequential(nn.Upsample(
        scale_factor=scale_factor, mode=mode
    ),)
    model.eval()

    input = torch.randn((1, inputs,  32, 32))
    expected = 0 if mode == 'nearest' else 11 * output_numel(model, input)
    check_macs(model, input, expected)
