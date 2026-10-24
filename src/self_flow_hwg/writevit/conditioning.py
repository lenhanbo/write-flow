from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .attention import CrossConditionBlock, PosCNN, TransformerBlock


class SpatialLayerNorm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, x.shape[1:])


class WriteViTWriterEncoder(nn.Module):
    """Writer feature extractor adapted from WriteViT's ``Writer`` module."""
    def __init__(self, dim: int = 256, depth: int = 3, heads: int = 4, output_tokens: int = 8):
        super().__init__()
        self.output_tokens = output_tokens
        self.layer_norm = SpatialLayerNorm()
        self.patch = nn.Conv2d(1, dim, kernel_size=8, stride=4, padding=2)
        self.pos_block = PosCNN(dim)
        self.downsample_blocks = nn.ModuleList([TransformerBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feature = self.patch(self.layer_norm(image))
        batch, channels, height, width = feature.shape
        tokens = feature.reshape(batch, channels, -1).transpose(1, 2)
        for index, block in enumerate(self.downsample_blocks):
            tokens = block(tokens)
            if index == 0:
                tokens = self.pos_block(tokens, height, width)
        tokens = self.norm(tokens)
        return F.adaptive_avg_pool1d(tokens.transpose(1, 2), self.output_tokens).transpose(1, 2)


def _load_unifont_table(path: str | Path, vocabulary: list[str]) -> torch.Tensor:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"WriteViT Unifont asset not found: {path}. Attach it as a Kaggle Dataset "
            "or set model.unifont_path to a valid local path."
        )
    with path.open("rb") as handle:
        symbols = pickle.load(handle)
    by_codepoint = {int(item["idx"][0]): np.asarray(item["mat"], dtype=np.float32).reshape(-1) for item in symbols}
    blank = np.zeros(256, dtype=np.float32)
    rows = [blank if token.startswith("<") or len(token) != 1 else by_codepoint.get(ord(token), blank) for token in vocabulary]
    return torch.from_numpy(np.stack(rows))


class UnifontQueryEncoder(nn.Module):
    """WriteViT Unifont embeddings with a learned fallback for portable smoke tests."""
    def __init__(self, vocabulary: list[str], dim: int, unifont_path: str | Path | None = None, use_unifont: bool | None = None):
        super().__init__()
        use_unifont = bool(unifont_path) if use_unifont is None else use_unifont
        if use_unifont:
            table = _load_unifont_table(unifont_path, vocabulary) if unifont_path else torch.zeros(len(vocabulary), 256)
            self.register_buffer("glyphs", table, persistent=True)
            self.learned = None
            self.linear = nn.Linear(table.shape[1], dim)
            self.uses_unifont = True
        else:
            self.register_buffer("glyphs", torch.empty(0), persistent=False)
            self.learned = nn.Embedding(len(vocabulary), dim, padding_idx=0)
            self.linear = nn.Identity()
            self.uses_unifont = False

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if self.learned is not None:
            return self.learned(ids)
        return self.linear(self.glyphs[ids])


class WriteViTConditioner(nn.Module):
    """Writer tokens + Unifont query tokens fused as in WriteViT, for a Flow DiT."""
    def __init__(
        self, vocabulary: list[str], output_dim: int, writevit_dim: int = 256,
        writer_depth: int = 3, fusion_depth: int = 5, heads: int = 4,
        style_tokens: int = 8, max_text_length: int = 32,
        unifont_path: str | Path | None = None,
        use_unifont: bool | None = None,
    ):
        super().__init__()
        self.max_text_length = max_text_length
        self.writer = WriteViTWriterEncoder(writevit_dim, writer_depth, heads, style_tokens)
        self.query = UnifontQueryEncoder(vocabulary, writevit_dim, unifont_path, use_unifont)
        self.query_position = nn.Parameter(torch.randn(1, max_text_length, writevit_dim) * 0.02)
        self.fusion = nn.ModuleList([CrossConditionBlock(writevit_dim, heads) for _ in range(fusion_depth)])
        self.content_projection = nn.Linear(writevit_dim, output_dim)
        self.style_projection = nn.Linear(writevit_dim, output_dim)
        self.content_type = nn.Parameter(torch.randn(1, 1, output_dim) * 0.02)
        self.style_type = nn.Parameter(torch.randn(1, 1, output_dim) * 0.02)

    @property
    def uses_unifont(self) -> bool:
        return self.query.uses_unifont

    def forward(self, ids: torch.Tensor, valid_mask: torch.Tensor, reference: torch.Tensor):
        if ids.shape[1] > self.max_text_length:
            raise ValueError(f"Text has {ids.shape[1]} tokens, maximum is {self.max_text_length}")
        style = self.writer(reference)
        content = self.query(ids) + self.query_position[:, : ids.shape[1]]
        for block in self.fusion:
            content = block(content, style)
        content = self.content_projection(content) + self.content_type
        style = self.style_projection(style) + self.style_type
        style_mask = torch.ones(style.shape[:2], dtype=torch.bool, device=style.device)
        return content, valid_mask, style, style_mask


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "netW"):
            if isinstance(checkpoint.get(key), dict):
                return checkpoint[key]
        if checkpoint and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
            return checkpoint
    raise ValueError("Unsupported WriteViT checkpoint format; expected a state_dict or model/netW mapping")


def load_writevit_weights(module: WriteViTConditioner, path: str | Path) -> dict[str, list[str]]:
    """Load reusable Writer/Unifont/CrossBlock weights from common WriteViT checkpoints."""
    state = _extract_state_dict(torch.load(path, map_location="cpu", weights_only=False))
    translated: dict[str, torch.Tensor] = {}
    prefixes = ("module.", "netW.", "netG.")
    for raw_key, value in state.items():
        key = raw_key
        while key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("netW."):
            key = "writer." + key[len("netW.") :]
        elif key.startswith("netG.query_embed.linear."):
            key = "query.linear." + key[len("netG.query_embed.linear.") :]
        elif key.startswith("netG.blocks_2."):
            key = "fusion." + key[len("netG.blocks_2.") :]
        elif key.startswith("conditioning."):
            key = key[len("conditioning.") :]
        elif key.startswith(("patch.", "pos_block.", "downsample_blocks.", "norm.")):
            key = "writer." + key
        elif key.startswith("query_embed.linear."):
            key = "query.linear." + key[len("query_embed.linear.") :]
        elif key.startswith("blocks_2."):
            key = "fusion." + key[len("blocks_2.") :]
        if key.startswith(("writer.", "query.linear.", "fusion.")):
            translated[key] = value
    expected = module.state_dict()
    compatible = {key: value for key, value in translated.items() if key in expected and expected[key].shape == value.shape}
    skipped = sorted(set(translated) - set(compatible))
    result = module.load_state_dict(compatible, strict=False)
    return {"missing": list(result.missing_keys), "unexpected": list(result.unexpected_keys), "loaded": sorted(compatible), "skipped": skipped}

