import torch

from self_flow_hwg.models.autoencoder import ContinuousAutoencoder, LatentStats, reconstruction_loss


def test_autoencoder_shapes_range_and_gradients():
    model = ContinuousAutoencoder(8, 8)
    x = torch.rand(2, 1, 64, 256)
    z = model.encode(x); prediction = model.decode(z)
    assert z.shape == (2, 8, 16, 64)
    assert prediction.shape == x.shape and prediction.min() >= 0 and prediction.max() <= 1
    loss, _ = reconstruction_loss(prediction, x, (x < .5).float()); loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    stats = LatentStats(z.mean((0, 2, 3), keepdim=True), z.std((0, 2, 3), keepdim=True))
    assert torch.allclose(stats.denormalize(stats.normalize(z)), z, atol=1e-5)

