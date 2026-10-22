# Conditional Self-Flow for handwriting generation

PyTorch research implementation for word-level offline handwriting synthesis conditioned on query text and a different word from the same writer. It uses a deterministic continuous autoencoder and rectified flow in latent space; there is no VQ/codebook.

The model combines:

- a character Transformer for content;
- a patch-token CNN style encoder inspired by WriteViT's writer encoder;
- a compact conditional DiT with a separate timestep for every image token;
- four controlled variants: `vanilla_fm`, `dual_fm`, `self_flow`, and the proposed `ink_self_flow` extension.

Self-Flow follows the clean-at-`t=0`, noise-at-`t=1` convention in [Chefer et al.](https://arxiv.org/abs/2603.06507). The official [Self-Flow repository](https://github.com/black-forest-labs/Self-Flow) currently exposes ImageNet inference; this repository implements the paper's training mechanism for handwriting and does not claim to reproduce its experiments. Ink-aware masking is an unvalidated extension, not part of the original paper. WriteViT is referenced architecturally; its GAN, OCR loss and required Unifont representation are intentionally not copied.

## Install

Python 3.10+ is required.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
pytest -q
```

All commands also work without installation by setting `$env:PYTHONPATH = "src"`.

## IAM preparation

Obtain IAM separately under its license. The root must contain `ascii/words.txt`, `ascii/forms.txt`, and `words/`. Writer IDs are read from `forms.txt`; they are never guessed from filenames. Segmentation-error rows and missing files are reported. Writers with fewer than two valid words are omitted because a target cannot serve as its own reference.

```powershell
python -m self_flow_hwg.cli.prepare_iam --iam-root "D:\data\IAM" --out "data\splits" --seed 42
python -m self_flow_hwg.cli.check_data --config "configs\self_flow_tiny.yaml" --output "outputs\data_check"
```

Preparation saves exact writer-disjoint JSON manifests, hashes, the train-only vocabulary, and a summary. Training references are sampled dynamically; validation/test pairing is seeded and fixed. Images retain aspect ratio on a white `64×256` canvas (`white=1`, `ink=0`).

## Train

Train one AE and reuse it for A–D:

```powershell
python -m self_flow_hwg.train.train_ae --config "configs\ae_tiny.yaml" --output "outputs\ae"
python -m self_flow_hwg.train.train_flow --config "configs\fm_tiny.yaml" --ae-checkpoint "outputs\ae\best.pt" --output "outputs\A_fm"
python -m self_flow_hwg.train.train_flow --config "configs\dual_fm_tiny.yaml" --ae-checkpoint "outputs\ae\best.pt" --output "outputs\B_dual"
python -m self_flow_hwg.train.train_flow --config "configs\self_flow_tiny.yaml" --ae-checkpoint "outputs\ae\best.pt" --output "outputs\C_self"
python -m self_flow_hwg.train.train_flow --config "configs\ink_self_flow_tiny.yaml" --ae-checkpoint "outputs\ae\best.pt" --output "outputs\D_ink"
```

Resume with `--resume "outputs\C_self\last.pt"` after increasing `train.max_steps`. Checkpoints contain model, EMA teacher where applicable, objective/projector, optimizer, AMP scaler, latent statistics/config/vocabulary, step and torch RNG state. JSONL losses and an AE reconstruction grid are written beside checkpoints.

For a 6 GB GPU, start with batch size 1–2, hidden dim 64–128, gradient accumulation 4–8, AMP `auto`, and activation checkpointing enabled. If CUDA runs out of memory, reduce batch size first, then hidden dimension/depth/style tokens. The tiny presets prioritize smoke testing, not quality.

## Synthetic end-to-end smoke test

No IAM files are needed:

```powershell
python -m self_flow_hwg.train.train_ae --config "configs\ae_tiny.yaml" --output "outputs\smoke_ae" --synthetic-smoke
python -m self_flow_hwg.train.train_flow --config "configs\self_flow_tiny.yaml" --ae-checkpoint "outputs\smoke_ae\best.pt" --output "outputs\smoke_flow" --synthetic-smoke
```

The smoke data validates execution and tensor contracts only; it cannot demonstrate handwriting quality.

## Generate and evaluate

```powershell
python -m self_flow_hwg.cli.sample --checkpoint "outputs\C_self\best.pt" --ae-checkpoint "outputs\ae\best.pt" --text "minimum" --reference "D:\data\reference.png" --seed 17 --steps 30 --output "outputs\samples\minimum.png"
python -m self_flow_hwg.cli.evaluate --config "configs\self_flow_tiny.yaml" --checkpoint "outputs\C_self\best.pt" --ae-checkpoint "outputs\ae\best.pt" --output "outputs\C_self\eval"
```

Sampling integrates the ODE backward from `t=1` to `t=0` with Euler or Heun and uses homogeneous timesteps; the training foreground map and teacher are not inference inputs. External CER/WER and style scores are explicitly reported as unavailable unless compatible frozen weights are supplied. Record real experiments in [RESULTS.md](RESULTS.md).

## What is tested

Automated tests cover writer-disjoint splitting, non-self references, vocabulary and padding, grayscale masks, AE shapes/range/gradients, lossless patch inversion, per-token time conditioning, shared-noise dual scheduling, teacher time, exact ink-aware mask cardinality, frozen EMA behavior, all four objectives, and backward ODE sign/determinism.

Dataset-dependent quality, external recognizer/style metrics, and GPU memory fit cannot be asserted without user-supplied IAM data, weights, and hardware.
"# write-flow" 
