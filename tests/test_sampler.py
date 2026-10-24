import torch

from self_flow_hwg.flow.sampler import ode_sample


def test_backward_euler_constant_velocity_and_determinism():
    x = torch.ones(1, 2, 3)
    velocity = lambda value, tau: torch.full_like(value, 2)
    assert torch.allclose(ode_sample(velocity, x, steps=4), torch.full_like(x, -1))
    assert torch.equal(ode_sample(velocity, x, steps=3), ode_sample(velocity, x, steps=3))

