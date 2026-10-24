import torch

from self_flow_hwg.models.dit import ConditionalDiT, patchify, unpatchify
from self_flow_hwg.models.timestep import PerTokenTimeEmbedding


def test_patch_round_trip_and_token_order():
    x = torch.randn(2, 8, 16, 64)
    assert torch.equal(unpatchify(patchify(x, 2), 8, 16, 64, 2), x)


def test_per_token_time_changes_corresponding_embedding():
    module = PerTokenTimeEmbedding(16, 8).eval()
    base = torch.zeros(1, 4); changed = base.clone(); changed[0, 2] = .7
    a, b = module(base), module(changed)
    assert torch.allclose(a[:, [0, 1, 3]], b[:, [0, 1, 3]])
    assert not torch.allclose(a[:, 2], b[:, 2])


def test_dit_output_and_hidden_shapes():
    model = ConditionalDiT(2, (4, 8), 2, 16, 2, 4)
    x = torch.randn(2, 8, 8); tau = torch.rand(2, 8)
    cond = torch.randn(2, 3, 16); mask = torch.ones(2, 3, dtype=torch.bool)
    out, hidden = model(x, tau, cond, mask, cond[:, :1], mask[:, :1], (0,))
    assert out.shape == x.shape and hidden[0].shape == (2, 8, 16)

