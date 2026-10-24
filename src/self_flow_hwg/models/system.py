from __future__ import annotations

import torch
from torch import nn

from .content_encoder import ContentEncoder
from .dit import ConditionalDiT
from .style_encoder import StyleEncoder
from self_flow_hwg.writevit import WriteViTConditioner, load_writevit_weights


class ConditionalFlowModel(nn.Module):
    def __init__(self, vocabulary: int | list[str], config: dict):
        super().__init__()
        vocab_size = vocabulary if isinstance(vocabulary, int) else len(vocabulary)
        vocab_items = [str(i) for i in range(vocab_size)] if isinstance(vocabulary, int) else vocabulary
        dim = int(config.get("hidden_dim", 256))
        self.conditioning_type = str(config.get("conditioning", "native"))
        self.conditioning_load_report: dict[str, list[str]] | None = None
        if self.conditioning_type == "writevit":
            self.conditioning = WriteViTConditioner(
                vocab_items, dim, int(config.get("writevit_dim", 256)),
                int(config.get("writer_depth", 3)), int(config.get("fusion_depth", 5)),
                int(config.get("writevit_heads", 4)), int(config.get("style_tokens", 8)),
                int(config.get("max_text_length", 32)), config.get("unifont_path"), config.get("use_unifont"),
            )
            checkpoint = config.get("writevit_checkpoint")
            if checkpoint:
                self.conditioning_load_report = load_writevit_weights(self.conditioning, checkpoint)
            if bool(config.get("freeze_conditioning", False)):
                self.conditioning.requires_grad_(False)
            self.content = None
            self.style = None
        elif self.conditioning_type == "native":
            self.conditioning = None
            self.content = ContentEncoder(vocab_size, dim, int(config.get("max_text_length", 32)), int(config.get("content_depth", 2)), int(config.get("num_heads", 4)))
            self.style = StyleEncoder(dim, int(config.get("style_tokens", 8)))
        else:
            raise ValueError(f"Unknown conditioning={self.conditioning_type!r}; expected 'native' or 'writevit'")
        self.dit = ConditionalDiT(
            int(config.get("ae_latent_channels", 8)), tuple(config.get("latent_size", [16, 64])),
            int(config.get("patch_size", 2)), dim, int(config.get("depth", 8)), int(config.get("num_heads", 4)), bool(config.get("activation_checkpointing", False)),
        )

    def conditions(self, text_ids: torch.Tensor, text_mask: torch.Tensor, reference: torch.Tensor):
        if self.conditioning is not None:
            return self.conditioning(text_ids, text_mask, reference)
        content, content_mask = self.content(text_ids, text_mask)
        style, style_mask = self.style(reference)
        return content, content_mask, style, style_mask

