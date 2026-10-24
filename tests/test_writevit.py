import pickle

import numpy as np
import torch

from self_flow_hwg.writevit.conditioning import WriteViTConditioner, load_writevit_weights


def test_writevit_conditioner_shapes_and_gradients():
    vocabulary = ["<pad>", "<unk>", "a", "b"]
    model = WriteViTConditioner(vocabulary, output_dim=32, writevit_dim=32, writer_depth=1, fusion_depth=1, heads=4, style_tokens=4)
    ids = torch.tensor([[2, 3], [3, 0]])
    mask = ids != 0
    content, content_mask, style, style_mask = model(ids, mask, torch.rand(2, 1, 64, 256))
    assert content.shape == (2, 2, 32)
    assert style.shape == (2, 4, 32)
    assert torch.equal(content_mask, mask) and style_mask.all()
    (content.mean() + style.mean()).backward()
    assert model.writer.patch.weight.grad is not None


def test_unifont_asset_and_checkpoint_loading(tmp_path):
    symbols = [
        {"idx": [ord("a")], "mat": np.ones((16, 16), dtype=np.float32)},
        {"idx": [ord("b")], "mat": np.full((16, 16), 2, dtype=np.float32)},
    ]
    asset = tmp_path / "unifont.pickle"
    with asset.open("wb") as handle:
        pickle.dump(symbols, handle)
    model = WriteViTConditioner(["<pad>", "<unk>", "a", "b"], 32, writevit_dim=32, writer_depth=1, fusion_depth=1, heads=4, style_tokens=2, unifont_path=asset)
    assert model.uses_unifont and model.query.glyphs.shape == (4, 256)
    portable = WriteViTConditioner(["<pad>", "<unk>", "a", "b"], 32, writevit_dim=32, writer_depth=1, fusion_depth=1, heads=4, style_tokens=2, use_unifont=True)
    portable.load_state_dict(model.state_dict())
    assert torch.equal(portable.query.glyphs, model.query.glyphs)
    checkpoint = tmp_path / "writevit.pt"
    expected = torch.randn_like(model.writer.patch.weight)
    torch.save({"netW.patch.weight": expected}, checkpoint)
    report = load_writevit_weights(model, checkpoint)
    assert "writer.patch.weight" in report["loaded"]
    assert torch.equal(model.writer.patch.weight, expected)
