"""Portable WriteViT-derived conditioning components.

Adapted from DAIR-Group/WriteViT (MIT License). The GAN generator,
discriminator, OCR, and writer-classification training are intentionally absent.
"""

from .conditioning import WriteViTConditioner, load_writevit_weights

__all__ = ["WriteViTConditioner", "load_writevit_weights"]

