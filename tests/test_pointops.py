import torch
import pytest
from functions.pointops import (
    furthestsampling,
    knnquery,
    grouping,
    subtraction,
    aggregation,
    interpolation,
    queryandgroup,
)


def test_furthestsampling_square():
    # 4 points in a square, sample 2: expect [0,3]
    xyz = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=torch.float,
    )
    offset = torch.tensor([4], dtype=torch.int)
    new_offset = torch.tensor([2], dtype=torch.int)
    idx = furthestsampling(xyz, offset, new_offset)
    assert list(idx.shape) == [2]
    # first always start index 0, second farthest is point 3
    assert idx[0].item() == 0
    assert idx[1].item() == 3


def test_knnquery_vs_torch():
    # Compare to torch.cdist + topk
    n, k = 10, 3
    xyz = torch.randn(n, 3)
    offset = torch.tensor([n], dtype=torch.int)
    new_offset = torch.tensor([n], dtype=torch.int)
    idx, dist = knnquery(k, xyz, None, offset, new_offset)
    assert idx.shape == (n, k)
    assert dist.shape == (n, k)
    # brute-force
    d = torch.cdist(xyz, xyz)
    dist0, idx0 = torch.topk(d, k, largest=False, sorted=True)
    # compare indices and distances
    assert torch.equal(idx, idx0)
    assert torch.allclose(dist, dist0, atol=1e-6)


def test_grouping_simple():
    # input shape (6,2), idx shape (2,2)
    feat = torch.arange(12, dtype=torch.float).view(6, 2)
    idx = torch.tensor([[0, 4], [5, 1]], dtype=torch.int)
    out = grouping(feat, idx)
    assert out.shape == (2, 2, 2)
    # manual gather
    expected = torch.stack([feat[idx[0]], feat[idx[1]]], dim=0)
    assert torch.equal(out, expected)


def test_subtraction_simple():
    # input1, input2 shape (3,2), idx shape (3,2)
    input1 = torch.arange(6, dtype=torch.float).view(3, 2)
    input2 = torch.arange(6, 0, -1, dtype=torch.float).view(3, 2)
    idx = torch.tensor([[0, 2], [1, 2], [2, 0]], dtype=torch.int)
    out = subtraction(input1, input2, idx)
    assert out.shape == (3, 2, 2)
    # manual compute
    expected = torch.zeros_like(out)
    for i in range(3):
        for j in range(2):
            expected[i, j] = input1[i] - input2[idx[i, j]]
    assert torch.equal(out, expected)


def test_interpolation_linear():
    # two points on x-axis, linear interpolation
    xyz = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float)
    new_xyz = torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float)
    feat = torch.tensor([[0.0], [2.0]], dtype=torch.float)
    offset = torch.tensor([2], dtype=torch.int)
    new_offset = torch.tensor([1], dtype=torch.int)
    out = interpolation(xyz, new_xyz, feat, offset, new_offset, k=2)
    # weights are both 0.5 -> 0*0.5 + 2*0.5 = 1
    assert out.shape == (1, 1)
    assert torch.allclose(out, torch.tensor([[1.0]]), atol=1e-6)


def test_aggregation_simple():
    # n=1, nsample=1, c=1, w_c=1
    inp = torch.tensor([[2.0]], dtype=torch.float)
    pos = torch.tensor([[[3.0]]], dtype=torch.float)
    wt = torch.tensor([[[4.0]]], dtype=torch.float)
    idx = torch.tensor([[0]], dtype=torch.int)
    out = aggregation(inp, pos, wt, idx)
    # (2+3)*4 = 20
    assert out.shape == (1, 1)
    assert torch.allclose(out, torch.tensor([[20.0]]), atol=1e-6)


def test_queryandgroup_default():
    # with nsample=1, use_xyz=False, should gather own feature
    xyz = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float)
    feat = torch.tensor([[1.0], [2.0]], dtype=torch.float)
    offset = torch.tensor([2], dtype=torch.int)
    new_offset = torch.tensor([2], dtype=torch.int)
    out = queryandgroup(1, xyz, None, feat, None, offset, new_offset, use_xyz=False)
    # shape (2,1,1)
    assert out.shape == (2, 1, 1)
    assert torch.equal(out[:, 0, 0], feat.squeeze())
