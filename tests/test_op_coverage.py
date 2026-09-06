import torch
from torch import nn
import pytest
import tempfile
import os
from onnx_opcounter import calculate_macs
from onnx_opcounter.onnx_opcounter import FREE_OPS, ELEMENTWISE_OPS, REDUCTION_OPS
import onnx


def export_onnx(model, args, opset=16):
    model.eval()
    if not isinstance(args, tuple):
        args = (args,)

    tmp = tempfile.TemporaryDirectory()
    path = os.path.join(tmp.name, "_model.onnx")
    torch.onnx.export(model, args, path, output_names=['output'],
                      opset_version=opset, dynamo=False)
    try:
        return onnx.load_model(path)
    finally:
        tmp.cleanup()


def macs_of(model, args, opset=16):
    return calculate_macs(export_onnx(model, args, opset))


class Lambda(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *args):
        return self.fn(*args)


@pytest.mark.parametrize('kernel, stride, padding', [(2, 2, 0), (3, 2, 1), (3, 1, 1)])
@pytest.mark.parametrize('pool', ['max', 'avg'])
def test_pool_counts_the_kernel_window(kernel, stride, padding, pool):
    channels, size = 16, 32
    layer = (nn.MaxPool2d if pool == 'max' else nn.AvgPool2d)(
        kernel, stride=stride, padding=padding)
    model = nn.Sequential(layer)

    x = torch.randn(1, channels, size, size)
    with torch.no_grad():
        out_elements = model(x).numel()

    assert macs_of(model, x) == out_elements * kernel * kernel


@pytest.mark.parametrize('channels, size', [(1, 8), (16, 32), (64, 7)])
@pytest.mark.parametrize('pool', ['avg', 'max'])
def test_global_pool_reads_every_input_element(channels, size, pool):
    layer = nn.AdaptiveAvgPool2d(1) if pool == 'avg' else nn.AdaptiveMaxPool2d(1)
    x = torch.randn(1, channels, size, size)

    assert macs_of(nn.Sequential(layer), x) == channels * size * size


class Recurrent(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        return self.layer(x)[0]


@pytest.mark.parametrize('inputs, hidden, seq', [(8, 16, 5), (256, 512, 20)])
@pytest.mark.parametrize('cell, gates', [('lstm', 4), ('gru', 3), ('rnn', 1)])
def test_recurrent_scales_with_gates_and_timesteps(inputs, hidden, seq, cell, gates):
    layer = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}[cell]
    model = Recurrent(layer(inputs, hidden, batch_first=True))

    x = torch.randn(1, seq, inputs)
    expected = seq * gates * hidden * (inputs + hidden)

    assert macs_of(model, x) == expected


def test_recurrent_is_not_charged_as_its_output_size():
    inputs, hidden, seq = 256, 512, 100
    model = Recurrent(nn.LSTM(inputs, hidden, batch_first=True))

    counted = macs_of(model, torch.randn(1, seq, inputs))
    assert counted == seq * 4 * hidden * (inputs + hidden)
    assert counted > 1000 * (seq * hidden)


@pytest.mark.parametrize('name, fn, args', [
    ('Concat', lambda x: torch.cat([x, x], dim=1), (torch.randn(1, 16, 8, 8),)),
    ('Pad', lambda x: torch.nn.functional.pad(x, (1, 1, 1, 1)), (torch.randn(1, 16, 8, 8),)),
    ('Flatten', lambda x: x.flatten(1), (torch.randn(1, 16, 8, 8),)),
    ('PixelShuffle', lambda x: torch.nn.functional.pixel_shuffle(x, 2), (torch.randn(1, 16, 8, 8),)),
])
def test_data_movement_is_free(name, fn, args):
    assert macs_of(Lambda(fn), args) == 0


def test_expand_does_not_charge_for_the_broadcast_result():
    x = torch.randn(1, 16, 8, 8)
    counted = macs_of(Lambda(lambda t: t.expand(4, 16, 8, 8)), x)

    assert counted < 4 * 16 * 8 * 8 // 100


def test_broadcast_add_does_not_depend_on_operand_order():
    shape = (1, 32, 64)
    bias = torch.randn(64)
    elements = 1 * 32 * 64

    lhs = macs_of(Lambda(lambda x: bias + x), torch.randn(shape))
    rhs = macs_of(Lambda(lambda x: x + bias), torch.randn(shape))

    assert lhs == rhs == elements


def test_reduction_reads_its_whole_input():
    x = torch.randn(1, 32, 64)
    assert macs_of(Lambda(lambda t: t.sum()), x) == 32 * 64


def test_layernorm_matches_its_own_decomposition():
    model = nn.Sequential(nn.LayerNorm(32))
    x = torch.randn(1, 8, 32)

    decomposed = macs_of(model, x, opset=16)
    fused = macs_of(model, x, opset=17)

    assert fused == pytest.approx(decomposed, rel=0.02)
    assert fused == 1 * 8 * 32 * (5 + 2)


def test_layernorm_charges_the_affine_tensors_it_is_given():
    model = nn.Sequential(nn.LayerNorm(32, elementwise_affine=False))
    x = torch.randn(1, 8, 32)

    fused = export_onnx(model, (x,), opset=17)
    node = next(n for n in fused.graph.node if n.op_type == 'LayerNormalization')
    assert len(node.input) == 3
    assert calculate_macs(fused) == 1 * 8 * 32 * (5 + 2)

    assert macs_of(model, x, opset=16) == pytest.approx(1 * 8 * 32 * 5, rel=0.02)


def test_batchnorm_stays_a_plain_affine_transform():
    x = torch.randn(1, 16, 8, 8)
    assert macs_of(nn.Sequential(nn.BatchNorm2d(16)), x) == 2 * 16 * 8 * 8


@pytest.mark.parametrize('equation, shapes, expected', [
    ('bik,bkj->bij', [(2, 8, 16), (2, 16, 32)], 2 * 8 * 32 * 16),
    ('ij,jk->ik', [(4, 8), (8, 16)], 4 * 16 * 8),
    ('bhqd,bhkd->bhqk', [(1, 4, 8, 16), (1, 4, 8, 16)], 1 * 4 * 8 * 8 * 16),
])
def test_einsum_charges_the_contracted_dimensions(equation, shapes, expected):
    model = Lambda(lambda a, b: torch.einsum(equation, a, b))
    args = tuple(torch.randn(s) for s in shapes)

    assert macs_of(model, args) == expected


def test_every_standard_onnx_op_is_classified():
    import onnx.defs
    import re

    source = open(os.path.join(os.path.dirname(__file__), os.pardir,
                               'onnx_opcounter', 'onnx_opcounter.py')).read()
    explicit = set(re.findall(r"^\s+'(\w+)': \w+_macs,", source, re.M))
    classified = FREE_OPS | ELEMENTWISE_OPS | REDUCTION_OPS | explicit

    standard = {s.name for s in onnx.defs.get_all_schemas()
                if s.domain == '' and not s.deprecated}

    assert not standard - classified


def test_op_categories_are_disjoint():
    assert not FREE_OPS & ELEMENTWISE_OPS
    assert not FREE_OPS & REDUCTION_OPS
    assert not ELEMENTWISE_OPS & REDUCTION_OPS
