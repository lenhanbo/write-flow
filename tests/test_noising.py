import torch

from self_flow_hwg.flow.mask_samplers import ink_aware_mask, token_ink_occupancy
from self_flow_hwg.flow.noising import dual_timestep_batch, interpolate


def test_endpoints_and_shared_noise_teacher_is_cleaner():
    clean = torch.randn(2, 6, 3); noise = torch.randn_like(clean)
    assert torch.equal(interpolate(clean, noise, torch.zeros(2, 6)), clean)
    assert torch.equal(interpolate(clean, noise, torch.ones(2, 6)), noise)
    batch = dual_timestep_batch(clean, torch.tensor([[1, 0, 0, 1, 0, 0], [0, 1, 0, 0, 1, 0]], dtype=torch.bool))
    assert torch.equal(batch.teacher_tau, torch.minimum(batch.t, batch.s).expand_as(batch.teacher_tau))
    assert torch.allclose(batch.target_velocity, batch.noise - clean)


def test_ink_mask_exact_unique_and_background_fallback():
    fg = torch.zeros(2, 1, 64, 256); fg[0, :, :8, :8] = 1
    occupancy = token_ink_occupancy(fg, (8, 32)); mask = ink_aware_mask(occupancy, .25)
    assert mask.shape == (2, 256)
    assert torch.equal(mask.sum(1), torch.tensor([64, 64]))

