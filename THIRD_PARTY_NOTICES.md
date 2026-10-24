# Third-party notices

Parts of `src/self_flow_hwg/writevit/` are adapted from
[DAIR-Group/WriteViT](https://github.com/DAIR-Group/WriteViT), licensed under the
MIT License, Copyright (c) 2025 Đặng Hoài Nam.

The adaptation removes GAN/OCR components, replaces the `timm` dependency with
PyTorch-only blocks, makes Unifont assets configurable, and exposes writer/query
tokens as conditioning for a latent Flow Transformer.

The original MIT license text is reproduced in `licenses/WRITEVIT_LICENSE`.

