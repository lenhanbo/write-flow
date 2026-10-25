# Conditional Self-Flow for handwriting generation

PyTorch research implementation for word-level offline handwriting synthesis conditioned on query text and a different word from the same writer. It uses a deterministic continuous autoencoder and rectified flow in latent space; there is no VQ/codebook.

The model combines:

- a character Transformer for content;
- either native lightweight conditioning or a portable WriteViT Writer + Unifont + cross-attention conditioner;
- a compact conditional DiT with a separate timestep for every image token;
- four controlled variants: `vanilla_fm`, `dual_fm`, `self_flow`, and the proposed `ink_self_flow` extension.

Self-Flow follows the clean-at-`t=0`, noise-at-`t=1` convention in [Chefer et al.](https://arxiv.org/abs/2603.06507). The official [Self-Flow repository](https://github.com/black-forest-labs/Self-Flow) currently exposes ImageNet inference; this repository implements the paper's training mechanism for handwriting and does not claim to reproduce its experiments. Ink-aware masking is an unvalidated extension, not part of the original paper. The optional WriteViT conditioner ports the Writer Encoder, Unifont query representation, and query-to-style CrossBlocks; its GAN generator/discriminator and OCR training are not used. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

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

Preparation saves exact writer-disjoint JSON manifests, hashes, the train-only vocabulary, and a summary. Image paths inside new manifests are relative to `data.iam_root`, so the same manifests work on Windows and Kaggle. Training references are sampled dynamically; validation/test pairing is seeded and fixed. Images retain aspect ratio on a white `64×256` canvas (`white=1`, `ink=0`).

The local dataset at `../data/iam_dataset` uses a packaged layout with
`train_gt.txt`, `val_gt.txt`, and `words/` rather than IAM's `ascii/forms.txt`.
Prepare it with:

```powershell
python -m self_flow_hwg.cli.prepare_iam --iam-root "..\data\iam_dataset" --out "data\splits_local" --seed 42
python -m self_flow_hwg.cli.check_data --config "configs\writevit_fm_local.yaml" --output "outputs\local_data_check"
```

Because that package omits the official form-to-writer mapping, the loader uses
each form as a conservative style group. Words paired within a form necessarily
share a writer, but separate forms from the same writer are not merged. Any
form occurring in both supplied splits is removed from the held-out split.

## Train

Train one AE and reuse it for A–D:

```powershell
python -m self_flow_hwg.train.train_ae --config "configs\ae_tiny.yaml" --output "outputs\ae"
python -m self_flow_hwg.train.train_flow --config "configs\fm_tiny.yaml" --ae-checkpoint "outputs\ae\best.pt" --output "outputs\A_fm"
python -m self_flow_hwg.train.train_flow --config "configs\dual_fm_tiny.yaml" --ae-checkpoint "outputs\ae\best.pt" --output "outputs\B_dual"
python -m self_flow_hwg.train.train_flow --config "configs\self_flow_tiny.yaml" --ae-checkpoint "outputs\ae\best.pt" --output "outputs\C_self"
python -m self_flow_hwg.train.train_flow --config "configs\ink_self_flow_tiny.yaml" --ae-checkpoint "outputs\ae\best.pt" --output "outputs\D_ink"
```

## WriteViT-conditioned Flow

Set `model.conditioning: writevit` to replace the native content/style encoders with:

```text
reference -> WriteViT Writer Encoder -> style tokens
query IDs -> WriteViT Unifont Encoder -> query tokens
query tokens cross-attend to style tokens
styled query + style tokens -> latent Flow DiT
```

`configs/writevit_self_flow_tiny.yaml` uses learned character embeddings when `unifont_path` is null so unit and smoke tests remain self-contained. To train from scratch while retaining the WriteViT design, use the glyph table but leave the checkpoint empty:

```yaml
model:
  conditioning: writevit
  unifont_path: D:/datasets/writevit/unifont.pickle
  writevit_checkpoint: null
  freeze_conditioning: false
```

Then run:

```powershell
python -m self_flow_hwg.train.train_flow --config "configs\writevit_self_flow_tiny.yaml" --ae-checkpoint "outputs\ae\best.pt" --output "outputs\writevit_self_flow"
```

The Writer encoder, query/style CrossBlocks, and Flow network are initialized and trained together from scratch. `unifont.pickle` contains character bitmaps rather than trained weights; setting it to null switches to learned embeddings. Legacy checkpoint loading remains available only for optional ablations.

Start with the original-style pixel-space Flow Matching baseline before enabling latent Self-Flow:

```powershell
python -m self_flow_hwg.train.train_pixel_flow --config "configs\writevit_fm_tiny.yaml" --output "outputs\writevit_pixel_fm"
```

This baseline follows the convention of Lipman et al.: Gaussian pixel noise at `t=0`, image data at `t=1`, and a conditional OT probability path with configurable `sigma_min`. It constructs no autoencoder, latent normalization, EMA teacher, dual-timestep mask, representation projector, or ink-aware loss. Images are mapped from `[0,1]` to `[-1,1]` before Flow Matching.

Sample directly in pixel space—no AE checkpoint is needed:

```powershell
python -m self_flow_hwg.cli.sample_pixel_flow --checkpoint "outputs\writevit_pixel_fm\best.pt" --text "minimum" --reference "D:\data\reference.png" --seed 17 --steps 50 --output "outputs\samples\minimum_pixel_fm.png"
```

The older `train_flow` entry point remains available for separate latent FM/Self-Flow experiments and deliberately uses a frozen autoencoder.

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

## Kaggle

For the current baseline, attach only the original IAM dataset and use
`configs/writevit_fm_kaggle.yaml`. All outputs go to `/kaggle/working`.
Additional assets and an AE are only relevant to optional later experiments.
Copy-ready notebook cells are in [kaggle/README.md](kaggle/README.md). The
project does not initialize or modify a Git repository.

For the primary pixel-space FM baseline, `configs/writevit_fm_kaggle.yaml`
requires only the original IAM dataset. It creates splits under
`/kaggle/working/splits`, uses learned query embeddings, and trains every model
component from scratch.
"# write-flow" 
