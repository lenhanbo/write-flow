import torch

from self_flow_hwg.flow.pixel import pixel_ode_sample, pixel_ot_batch
from self_flow_hwg.models.pixel_flow import PixelFlowUNet, WriteViTPixelFlow


def test_pixel_ot_path_and_velocity():
    data = torch.randn(3, 1, 8, 16)
    batch = pixel_ot_batch(data, sigma_min=.01, generator=torch.Generator().manual_seed(4))
    weight = batch.timestep[:, None, None, None]
    expected = (1 - .99 * weight) * batch.noise + weight * data
    assert torch.allclose(batch.state, expected)
    assert torch.allclose(batch.velocity, data - .99 * batch.noise)


def test_pixel_sampler_runs_forward_from_zero_to_one():
    noise = torch.ones(1, 1, 4, 4)
    velocity = lambda state, timestep: torch.full_like(state, 2)
    result = pixel_ode_sample(velocity, noise, steps=4)
    assert torch.allclose(result, torch.full_like(noise, 3))


def test_pixel_unet_and_writevit_conditioning_gradients():
    unet = PixelFlowUNet(base_channels=8, condition_dim=32, heads=4)
    image = torch.randn(2, 1, 16, 32)
    content = torch.randn(2, 3, 32)
    style = torch.randn(2, 2, 32)
    content_mask = torch.ones(2, 3, dtype=torch.bool)
    style_mask = torch.ones(2, 2, dtype=torch.bool)
    output = unet(image, torch.rand(2), content, content_mask, style, style_mask)
    assert output.shape == image.shape

    config = {
        "condition_dim": 32, "base_channels": 8, "flow_heads": 4,
        "writevit_dim": 32, "writer_depth": 1, "fusion_depth": 1,
        "writevit_heads": 4, "style_tokens": 2, "max_text_length": 8,
    }
    model = WriteViTPixelFlow(["<pad>", "<unk>", "a", "b"], config)
    ids = torch.tensor([[2, 3], [3, 2]])
    prediction = model(image, torch.rand(2), ids, torch.ones_like(ids, dtype=torch.bool), torch.rand(2, 1, 16, 32))
    prediction.square().mean().backward()
    assert prediction.shape == image.shape
    assert model.flow.input.weight.grad is not None
    assert model.conditioning.writer.patch.weight.grad is not None
