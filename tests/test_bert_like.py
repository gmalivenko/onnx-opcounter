import math
import torch
from torch import nn
import pytest
import tempfile
import os
from onnx_opcounter import calculate_params, calculate_macs
import onnx


class SelfAttention(nn.Module):
    def __init__(self, hidden, heads):
        super().__init__()
        self.heads = heads
        self.head_dim = hidden // heads
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, hidden)

    def split_heads(self, x):
        batch, seq, _ = x.shape
        return x.view(batch, seq, self.heads, self.head_dim).permute(0, 2, 1, 3)

    def forward(self, x):
        batch, seq, hidden = x.shape
        query = self.split_heads(self.query(x))
        key = self.split_heads(self.key(x))
        value = self.split_heads(self.value(x))

        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        context = torch.matmul(scores.softmax(-1), value)
        context = context.permute(0, 2, 1, 3).reshape(batch, seq, hidden)
        return self.out(context)


class EncoderLayer(nn.Module):
    def __init__(self, hidden, heads, ff):
        super().__init__()
        self.attention = SelfAttention(hidden, heads)
        self.attention_norm = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, ff)
        self.fc2 = nn.Linear(ff, hidden)
        self.output_norm = nn.LayerNorm(hidden)

    def forward(self, x):
        x = self.attention_norm(x + self.attention(x))
        hidden = self.fc2(torch.nn.functional.gelu(self.fc1(x)))
        return self.output_norm(x + hidden)


class BertLike(nn.Module):
    def __init__(self, vocab, hidden, heads, ff, layers, max_pos):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab, hidden)
        self.position_embedding = nn.Embedding(max_pos, hidden)
        self.embedding_norm = nn.LayerNorm(hidden)
        self.layers = nn.ModuleList(
            [EncoderLayer(hidden, heads, ff) for _ in range(layers)]
        )

    def forward(self, input_ids, position_ids):
        x = self.token_embedding(input_ids) + self.position_embedding(position_ids)
        x = self.embedding_norm(x)
        for layer in self.layers:
            x = layer(x)
        return x


VOCAB = 128
MAX_POS = 32


def build_bert(hidden, heads, ff, layers):
    torch.manual_seed(0)
    model = BertLike(VOCAB, hidden, heads, ff, layers, MAX_POS)
    with torch.no_grad():
        for p in model.parameters():
            p.normal_()
    model.eval()
    return model


def export_onnx(model, seq):
    input_ids = torch.randint(0, VOCAB, (1, seq))
    position_ids = torch.arange(seq).unsqueeze(0)

    tmp = tempfile.TemporaryDirectory()
    path = os.path.join(tmp.name, "_bert.onnx")
    torch.onnx.export(model, (input_ids, position_ids), path,
                      input_names=['input_ids', 'position_ids'],
                      output_names=['output'], opset_version=16, dynamo=False)
    try:
        return onnx.load_model(path)
    finally:
        tmp.cleanup()


def per_layer_matmul_macs(seq, hidden, heads, ff):
    head_dim = hidden // heads
    projections = 4 * seq * hidden * hidden
    attention = 2 * heads * seq * seq * head_dim
    feed_forward = 2 * seq * hidden * ff
    return projections + attention + feed_forward


CONFIGS = [
    (32, 4, 64),
    (64, 8, 128),
    (48, 3, 96),
]


@pytest.mark.parametrize('hidden, heads, ff', CONFIGS)
@pytest.mark.parametrize('layers', [0, 1, 2, 3])
@pytest.mark.parametrize('seq', [8, 16])
def test_bert_like_params(hidden, heads, ff, layers, seq):
    model = build_bert(hidden, heads, ff, layers)
    onnx_model = export_onnx(model, seq)

    expected = sum(p.numel() for p in model.parameters())
    assert calculate_params(onnx_model) == expected


@pytest.mark.parametrize('hidden, heads, ff', CONFIGS)
@pytest.mark.parametrize('seq', [8, 16])
def test_bert_like_macs_scale_linearly_with_depth(hidden, heads, ff, seq):
    macs = {
        layers: calculate_macs(export_onnx(build_bert(hidden, heads, ff, layers), seq))
        for layers in (0, 1, 2, 3)
    }

    deltas = [macs[n + 1] - macs[n] for n in (0, 1, 2)]
    assert deltas[0] == deltas[1] == deltas[2]


@pytest.mark.parametrize('hidden, heads, ff', CONFIGS)
@pytest.mark.parametrize('seq', [8, 16])
def test_bert_like_matmuls_dominate(hidden, heads, ff, seq):
    one = calculate_macs(export_onnx(build_bert(hidden, heads, ff, 1), seq))
    two = calculate_macs(export_onnx(build_bert(hidden, heads, ff, 2), seq))
    measured = two - one

    expected_matmuls = per_layer_matmul_macs(seq, hidden, heads, ff)
    assert measured >= expected_matmuls
    assert measured <= 1.25 * expected_matmuls


def test_bert_like_weight_dedup_is_free():
    torch.manual_seed(0)
    plain = BertLike(VOCAB, 32, 4, 64, 2, MAX_POS)
    plain.eval()

    deduped = export_onnx(plain, 8)
    distinct = export_onnx(build_bert(32, 4, 64, 2), 8)

    assert any(n.op_type == 'Identity' for n in deduped.graph.node)
    assert not any(n.op_type == 'Identity' for n in distinct.graph.node)

    assert calculate_macs(deduped) == calculate_macs(distinct)


def test_bert_like_embeddings_are_free():
    small = calculate_macs(export_onnx(build_bert(32, 4, 64, 0), 8))

    model = BertLike(VOCAB * 8, 32, 4, 64, 0, MAX_POS)
    torch.manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            p.normal_()
    model.eval()
    big_vocab = calculate_macs(export_onnx(model, 8))

    assert small == big_vocab
