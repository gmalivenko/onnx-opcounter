import numpy as np
import torch
from thop import profile
import pytest
from torch import nn
import tempfile
import os
from onnx_opcounter import calculate_macs
import onnx


def check_macs(model, input):
    macs, params = profile(model, inputs=(input,))

    with tempfile.TemporaryDirectory() as tmp:
        torch.onnx.export(model, input, os.path.join(tmp, "_model.onnx"),
                          verbose=True, input_names=['input'], output_names=['output'], opset_version=9)
        onnx_model = onnx.load_model(os.path.join(tmp, "_model.onnx"))
        onnx_macs = calculate_macs(onnx_model)
        print('macs', macs)
        print('onnx_macs', onnx_macs)
        assert int(macs) == int(onnx_macs)


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
    check_macs(model, input)


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
    check_macs(model, input)


@pytest.mark.parametrize('inputs', [1, 32, 64, 128, 256])
@pytest.mark.parametrize('outputs', [1, 32, 64, 128])
@pytest.mark.parametrize('bias', [True, False])
def test_linear_case1(inputs, outputs, bias):
    model = nn.Sequential(nn.Linear(
        inputs, outputs, bias=bias
    ),)
    model.eval()

    input = torch.randn((1, inputs))
    check_macs(model, input)


@pytest.mark.parametrize('inputs', [1, 32, 64, 128, 256])
@pytest.mark.parametrize('outputs', [1, 32, 64, 128])
@pytest.mark.parametrize('affine', [False])
def test_bn_case1(inputs, outputs, affine):
    model = nn.Sequential(nn.BatchNorm2d(
        inputs, outputs, affine=affine
    ))
    model.eval()

    input = torch.randn((1, inputs, 224, 224))
    check_macs(model, input)


@pytest.mark.parametrize('inputs', [1, 32, 64, 128, 256])
@pytest.mark.parametrize('scale_factor', [1, 2, 3, 5])
@pytest.mark.parametrize('mode', ['bilinear', 'nearest'])
# @pytest.mark.parametrize('mode', ['linear', 'nearest'])
def test_upsample_case1(inputs, scale_factor, mode):
    model = nn.Sequential(nn.Upsample(
        scale_factor=scale_factor, mode=mode
    ),)
    model.eval()

    input = torch.randn((1, inputs,  32, 32))
    check_macs(model, input)
#
# model = nn.Sequential(nn.Upsample(
#     scale_factor=4, mode='bilinear'
# ),)
# model.eval()
#
# input = torch.randn((1, 32,  224, 224))
# check_macs(model, input)
