# Codex Engineering Plan — Conditional Self-Flow for Offline Handwriting Generation

> **Use this file as the implementation brief for Codex.** Implement the project, not merely a paper summary. The goal is a reproducible, modular PyTorch research codebase for word-level offline handwriting image synthesis from (1) query text and (2) a separate same-writer reference image, using continuous latents and **Self-Supervised Flow Matching (Self-Flow)**. **Do not implement VQ or VQ-VAE.**
>
> Research reference: Hila Chefer et al., *Self-Supervised Flow Matching for Scalable Multi-Modal Synthesis*, ICML 2026, https://arxiv.org/html/2603.06507; official repository https://github.com/black-forest-labs/Self-Flow. The public Self-Flow repository primarily provides ImageNet inference code, not an end-to-end HWG training pipeline: implement training from the paper, do not claim a faithful reproduction of its original experimental setup.

## 0. Codex operating instructions

1. Inspect the current repository first. Reuse existing utilities where appropriate; do not delete or overwrite unrelated user code. If there is no project, create the structure specified below.
2. Implement in **five checkpoints** (P0–P4 below). Finish one checkpoint before moving to the next. At each checkpoint, run applicable automated tests and a **CPU or CUDA synthetic smoke test**, document the actual commands and outcomes, and fix failures. Do not fabricate success or metric scores.
3. Use **PyTorch**, Python >=3.10, torchvision/Pillow, PyYAML, tqdm, pytest; optional tensorboard, diffusers or torchmetrics only if necessary. Keep dependencies minimal; lock/configure them in `pyproject.toml` or `requirements.txt`. Support Windows paths, CPU, CUDA and optional BF16/FP16. Do not assume Linux-specific tooling.
4. The user has a **6 GB GPU**. Enable gradient accumulation, configurable activation checkpointing, frozen autoencoder during flow training, small-model presets, AMP, small batches and OOM advice. Hardware fit is not guaranteed; benchmark it.
5. Make **all experiment switches configuration-driven**. Vanilla FM, Dual-Timestep FM, Self-Flow and Ink-Aware Self-Flow must share architecture/data/optimizer by default for fair ablations.
6. Ensure all transformations and dimensions are explicit. Fail with informative messages if IAM images, metadata, writer IDs, or external evaluation weights are missing. **Do not auto-download IAM or pretend data is installed.**
7. Deliver working CLI entry points, unit tests, generated-sample contact sheets, a README, and a result table template. Avoid pseudocode in final code files.
8. Before finalizing, run formatting/static checks if configured, unit tests, and all synthetic smoke tests. Clearly report anything not exercised due to absent dataset, weights, or GPU.

## 1. Scope and non-goals

**In scope for MVP:** word-level, grayscale offline HWG; exact target transcription; reference image of same writer but a different word/image; continuous autoencoder; compact conditional DiT with per-image-token timestep conditioning; flow matching generation and Euler/Heun ODE sampling; optional EMA-teacher self-distillation; ink-aware schedule; controlled experiments.

**Non-goals for MVP:** VQ/codebooks, stroke trajectories, line-level generation, segmentation labels, stroke topology losses, adversarial training, expensive pretrained HTR dependency, GAN, complex diffusion pipeline, external teacher encoders, unconditional image generation.

### Data contract

Each example is a dict:

```python
{
    "target_image": FloatTensor[1, 64, 256],   # white ≈ 1, ink ≈ 0, [0,1]
    "reference_image": FloatTensor[1, 64, 256],
    "text_ids": LongTensor[L],                 # collated with pad/mask
    "text": str,                             # actual query/transcription
    "writer_id": str,
    "target_path": str,
    "reference_path": str,
    "foreground_mask": FloatTensor[1,64,256], # target mask; training only
}
```

Use a separate reference from the same writer and same split. Enforce `target_path != reference_path`. Default random reference can change between training iterations. For validation/test use a **fixed seeded reference choice** so comparisons are reproducible. Never use the target image or its duplicate as its own reference. The reference text need not equal target text and preferably differs. Keep writer-disjoint train/val/test where feasible; never silently leak writer IDs across partitions.

## 2. Repository layout

```text
self_flow_hwg/
├── README.md
├── pyproject.toml
├── .gitignore
├── configs/
│   ├── ae_tiny.yaml
│   ├── fm_tiny.yaml
│   ├── dual_fm_tiny.yaml
│   ├── self_flow_tiny.yaml
│   └── ink_self_flow_tiny.yaml
├── src/self_flow_hwg/
│   ├── __init__.py
│   ├── config.py
│   ├── data/
│   │   ├── iam.py
│   │   ├── preprocess.py
│   │   ├── pairing.py
│   │   └── collate.py
│   ├── models/
│   │   ├── autoencoder.py
│   │   ├── content_encoder.py
│   │   ├── style_encoder.py
│   │   ├── timestep.py
│   │   ├── dit.py
│   │   └── ema.py
│   ├── flow/
│   │   ├── noising.py
│   │   ├── mask_samplers.py
│   │   ├── losses.py
│   │   └── sampler.py
│   ├── train/
│   │   ├── train_ae.py
│   │   ├── train_flow.py
│   │   └── utils.py
│   ├── cli/
│   │   ├── prepare_iam.py
│   │   ├── check_data.py
│   │   ├── sample.py
│   │   └── evaluate.py
│   └── eval/
│       ├── recon.py
│       ├── text_metrics.py
│       ├── style_metrics.py
│       └── grids.py
├── tests/
│   ├── test_data.py
│   ├── test_autoencoder.py
│   ├── test_noising.py
│   ├── test_dit.py
│   ├── test_ema.py
│   ├── test_sampler.py
│   └── test_smoke_train.py
└── outputs/                          # gitignored
```

The importable package may also contain `__main__` wrappers or CLI functions. Exact file names may be adapted if an existing repository has established conventions, but retain modularity.

## 3. P0 — Data, preprocessing, and continuous autoencoder

### P0a. IAM word loader

- Accept user-specified IAM root and split files; use the official IAM offline **words** images and ASCII/XML writer metadata. Confirm mapping `word_id -> form_id -> writer_id`; do not guess writer IDs from filenames when metadata is ambiguous.
- Exclude samples annotated as segmentation errors (IAM words metadata status) and report counts and reasons. Handle apostrophes, punctuation and mixed case; construct character vocabulary **from train split only**, with `<pad>` and `<unk>`.
- Provide writer-disjoint split generation using a fixed seed if no trusted writer-disjoint split supplied. Save the actual writer lists and image lists to files; re-use exact split files in all runs. Check no overlap of writer IDs and image IDs. Some writers may have only one valid word; exclude from style-conditioned samples and log.
- Resize while preserving aspect ratio; fit into a fixed grayscale `64 x 256` canvas with white background, using configurable margins. **Do not stretch words horizontally**. Downscale too-wide words proportionately, log/optionally filter min-character-pixel-width and clipped samples. Avoid augmentations that change the transcription or make writer style unreliable.
- Compute a soft/binary target ink map (e.g., `foreground = clamp((1-gray)/threshold,0,1)` or configurable binary threshold); pad area has zero foreground. Do not make a target foreground map available to the generator at inference.
- Add `--dry-run` data checker showing sample counts, class/text length distribution, writer counts, fixed reference pairs, example images and selected masks.

### P0b. Small continuous image autoencoder (NO VQ)

- Implement `encode(image)->z`, `decode(z)->image`, downsampling **x4**. Starting input `[B,1,64,256]` -> latent `[B,8,16,64]` (configurable channels).
- Prefer a simple convolutional continuous AE for MVP; **do not call it a KL-VAE unless you implement a stochastic posterior and KL loss**. A KL-VAE can be an optional later replacement, but not a blocker.
- Encoder and decoder should use anti-aliased/strided convolutions and residual blocks as helpful; decoder output grayscale `[0,1]` (sigmoid). Train reconstruction with `L1 + λ_ink * foreground-weighted L1 + λ_edge * edge/gradient L1` (all configurable), avoiding extra heavy feature networks.
- Save checkpoint plus latent channel means/stds computed on **train data only**. Normalize encoded latent for flow training with saved `(z-mu)/sigma`, and reverse normalization before decoder. Guard zero std. Freeze AE for P1–P4. Ensure `z` scale is numerically stable; avoid using an uncalibrated Gaussian prior as if it were an AE posterior.
- Provide recon image grids plus foreground reconstruction errors and optional thresholded reconstruction CER when a valid recognizer is available. Gate P1 on visibly legible, stroke-preserving reconstruction; do not claim a fixed CER target.

**P0 acceptance:** reproducible writer-disjoint split, no self-reference, correct image range/shapes and masks, a trainable AE, valid latent round trip, recon grids and smoke test with synthetic strokes/images.

## 4. P1 — Vanilla Conditional Flow Matching baseline

### Network conditioning

- `ContentEncoder`: learned character embeddings + position embedding / small Transformer. Output `[B,L,D]` and boolean attention mask. Handle padding, max text length and unknown chars explicitly.
- `StyleEncoder`: compact CNN on `reference_image`, output `[B,K,D]` (e.g., pooled 4–8 style tokens) or one global style token. It must receive **only the reference**, not the target. In training, the style encoder is shared across runs and receives gradients via the generator.
- `ConditionalDiT`: patches **normalized AE latents** `[B,8,16,64]` into 2x2 latent patches, yielding `N=8*32=256` tokens of raw dimension 32. Use hidden dim ~256, 8 Transformer blocks, 4 or 8 heads (configurable); self-attention on image tokens, cross-attention to concatenated content/style tokens with padding masks. Positional encodings cover both latent axes. Predict **velocity** in token space, then unpatchify to latent shape. Expose hidden image-token states at selected intermediate layers.
- `PerTokenTimeEmbedding`: input `tau[B,N]`, output `[B,N,D]`, e.g., sinusoidal Fourier features and MLP. Apply per-token conditioning so every token actually responds to its own `tau_i`, not a broadcasted global embedding. Provide unit tests for this.
- Do not implement CFG in MVP; if implemented later, train classifier-free condition dropout and correctly handle the unconditional branch. Do not apply inference-only CFG without training for it.

### Flow formulation — IMPORTANT SIGN CONVENTION

Use Self-Flow paper convention: **clean at t=0, Gaussian noise at t=1**.

```python
z0 = normalize(encode(target_image))  # [B,C,H',W']
x0 = patchify(z0)                     # [B,N,P*P*C]
eps = torch.randn_like(x0)
t = torch.rand(B, 1, device=x0.device)
tau = t.expand(-1, N)
xt = (1 - tau[..., None]) * x0 + tau[..., None] * eps
v_target = eps - x0
v_pred, _ = dit(xt, tau, content_cond, style_cond)
loss_gen = ((v_pred - v_target) ** 2).mean()
```

- Sample ODE **backwards** from `t=1` noise to `t=0` data: `x_next = x + (t_next-t_current)*v_theta(x,t_current,cond)`, where `t_next-t_current < 0`. Provide Euler and optional Heun integrators (initial NFE 20/50).
- Sample should accept custom query string, arbitrary reference image, seed, number of ODE steps, output path, and checkpoint path. Decode only after inverse latent normalization; convert to `[0,1]` grayscale.
- During validation, use a frozen set of (`text`,`ref`,`noise_seed`) tuples and save grids including the input reference and query label.

**P1 acceptance:** single-batch forward/backward finite, gradients reach content/style encoder and generator, tiny synthetic overfit works qualitatively, training losses saved, generated samples produced with ODE sampler, same-seed deterministic sampling.

## 5. P2 — True Self-Flow (not just EMA weights)

### A. Dual-Timestep scheduling

- Draw **two independent timesteps per image** `t,s ~ Uniform(0,1)` (same distribution as vanilla FM). Draw a Bernoulli token mask `M[B,N]` with `ratio <= 0.5`, default `0.25`.
- Assign per-image-token tau: `tau_i = s` if masked, otherwise `t`.
- Use one **shared Gaussian noise realization** `eps` for Student and Teacher in the same step.

```python
t = torch.rand(B, 1, device=device)
s = torch.rand(B, 1, device=device)
M = torch.rand(B, N, device=device) < mask_ratio  # P2 random schedule
tau = torch.where(M, s.expand(-1,N), t.expand(-1,N))
x_student = (1-tau[...,None])*x0 + tau[...,None]*eps
cleaner_t = torch.minimum(t,s)
tau_teacher = cleaner_t.expand(-1,N)
x_teacher = (1-tau_teacher[...,None])*x0 + tau_teacher[...,None]*eps
```

**Key:** no token is zeroed out; the mask selects between time levels. Do not use a separate per-token iid timestep or hard inpainting mask. Teacher sees **all** tokens at `min(t,s)`.

### B. EMA teacher and representation alignment

- Initialize `teacher = deepcopy(student_generator)` including its compatible DiT backbone; freeze teacher parameters and set eval mode. EMA-update teacher generator parameters and buffers **after each optimizer step** (not every gradient accumulation micro-step). Suggested starting decay=0.9999, configurable, optionally with burn-in/warmup.
- Conditioning encoders are **shared** between Student and Teacher; compute `content_cond` and `style_cond` from online encoders. Teacher receives **detached conditions** and no gradients; Student gets normal gradients. EMA only replicates DiT, not the shared conditioning encoders. Document this as an adaptation and test it. Alternatively EMA-copy the entire conditioned generator if architecture makes that simpler, but NEVER mix inconsistent condition computations without documenting them.
- Take Student intermediate image-token features at a shallower block (8-block MVP: e.g., block 2) and Teacher at a deeper block (e.g., block 6). The head `projector` maps Student features to Teacher channel dimension (identity dimensions still use a learnable lightweight 2-layer MLP).
- Teacher forward must be under `torch.no_grad()`; teacher feature targets use stop-gradient. Compare aligned image-token positions with cosine similarity, average tokens and batch. Align hidden states **before velocity prediction**.
- Default `L_rep = 1 - mean(cosine(projector(h_student), stopgrad(h_teacher)))`. Paper writes negative cosine; these differ by an additive constant. Make gamma configurable, starting with 0.8 but use warmup/ramp-up to avoid unstable cold starts.
- Total `L = L_gen + gamma * L_rep`, keep loss terms and feature variances in logs.
- **P2A ablation**: `variant=dual_fm` uses dual timestep noising but `gamma=0`, no teacher forward.
- **P2B ablation**: `variant=self_flow` uses dual timestep + EMA/representation loss.
- EMA teacher is only a training mechanism. Inference uses the EMA student/generator weights if configured and all image tokens receive the same scalar time each ODE step. No teacher and no dual mask at inference.

**P2 acceptance:** per-token time embedding changes token activations, teacher parameters have no gradients, student/projector gradients exist, teacher EMA moves only after optimizer steps, teacher time equals `min(t,s)`, sampled images finite, cross-run same initial seeds/configs possible.

## 6. P3 — Ink-Aware Self-Flow (a research extension, NOT original Self-Flow)

- Compute target foreground occupancy per image token aligned to the AE+patch grid. With image size `64x256`, AE downsample 4x and DiT patch 2x2, each token corresponds to an `8x8` pixel block. Use average pooling / adaptive pooling to get `ink_occupancy[B,N] ∈ [0,1]`.
- Construct token sampling weight `w_i = alpha + beta * ink_occupancy_i`, e.g., `alpha=0.3, beta=0.7` to give nonzero probability to background tokens. Sample approximately/exactly `K=round(mask_ratio*N)` unique indices **without replacement**, with `torch.multinomial`; return boolean `M[B,N]`. Support `mask_mode = random | ink_aware`.
- Do **not** change `L_gen` or `L_rep` initially. Only change distribution of which tokens receive the second timestep; leave `t,s` distribution, teacher, and inference unchanged.
- Log mask coverage: fraction sampled foreground vs sampled background, mean ink occupancy, histograms and visualized overlays. Detect empty-ink samples and revert to uniform sampling when appropriate.
- This mask reads target foreground **only during training** and is never used as inference conditioning. Flag the risk of train–inference distribution shift: run exact ablations before presenting claims of improvement. No stroke annotations required.
- Add deterministic tests: all-background mask is valid, at least one ink token present case, unique sampled indices, exact tensor alignment, device compatibility, and no target-ink input to the generator.

**P3 acceptance:** same P2 quality controls, mask visualizations, `random` and `ink_aware` output equal shapes/valid per-token times, no leakage into sampling API.

## 7. P4 — Evaluation, experiment design and reproducibility

### Mandatory runs (same dataset, AE, architecture, budgets and seeds)

| Tag | variant | Timestep assignment | Self-distillation | Purpose |
|---|---|---|---|---|
| A | `vanilla_fm` | global single t | no | baseline |
| B | `dual_fm` | random two t,s | no | isolate scheduling |
| C | `self_flow` | random two t,s | yes | core Self-Flow |
| D | `ink_self_flow` | foreground-biased two t,s | yes | proposed extension |

Keep reference sampling and AE fixed across A–D. Run at least 3 seeds **when feasible**, report mean/std; never label single-run improvement statistically robust. Compare with and without EMA evaluation consistently (e.g., always EMA generator at eval, and use separately maintained eval EMA for baselines if needed).

### Measurements

1. **Content**: CER and WER via an explicitly documented pretrained handwriting text recognizer. Distinguish metrics on generated images vs autoencoder reconstructions, raw predictions vs normalized text. If recognition weights are not present, clearly output `unavailable`, not fabricated CER/WER. No recognition supervision is required for training the initial Self-Flow MVP.
2. **Style**: writer classification/retrieval similarity using a **held-out or frozen writer encoder** with documented training/split. Include reference-swap sanity test: same text/noise, different writers should show differences without changing the transcript. Recognizer/style encoder must not be trained on test writer labels.
3. **Image quality**: optional KID/FID/HWD only with consistent feature backbone, preprocessing and enough samples. Record sample counts and confidence where possible; small-sample FID is unstable. Always save grids for qualitative inspection.
4. **Reconstruction**: AE foreground/background MAE and edge difference; independently measure AE limitation before blaming the flow model.
5. **Efficiency**: params, train wall-time/step, GPU max allocated VRAM, throughput, ODE NFE and generation latency.

Save config snapshot, data split hashes, seed, git commit when available, logs (`JSONL`/TensorBoard), checkpoints (student, teacher, optimizer, scheduler, scaler, global step, RNG states, AE latent stats), best and last checkpoint, fixed-reference samples, eval results CSV/JSON. Resume must restore state reliably.

### Training caveats / hazards

- Grayscale convention: white=1, ink=0 throughout dataset. Convert only at AE normalization boundary; never accidentally reverse foreground.
- Mixed-precision AMP CPU/CUDA supported; use BF16 only when GPU supports it, otherwise FP16 with GradScaler or float32. Clip generator and encoder gradients; avoid updating EMA before optimizer step.
- Validate dimension math: `[B,1,64,256] -> [B,8,16,64] -> [B,256,32] -> [B,256,D]`. Must patchify/unpatchify losslessly for random tensors (same shapes and values).
- Inference ODE direction is **1 to 0**. All timesteps homogeneous at inference.
- `teacher.eval()`, `no_grad`, and consistent conditioning. No gradient into teacher. Student projection must be optimized; optional layer norm on projected features before cosine similarity.
- For some samples the mask can select both backgrounds and ink; do not confuse using a token's second timestep with fully masked (zeroed) data.
- Avoid simplistic interpretations: Self-Flow does not explicitly disentangle text/style or prove content correctness. Style encoder may be ignored; measure reference sensitivity and consider auxiliary writer consistency objectives only **after** baseline ablations.
- The official release is an ImageNet **inference** baseline, not an implementation of the whole proposed HWG pipeline. Cite any borrowed code and preserve license.

## 8. Test plan (MUST implement as automated tests)

- **Data**: no train/val/test writer overlap; no target==reference; vocabulary/pad masks; grayscale/canvas/ink occupancy; expected exclusion rules.
- **AE**: expected tensor shapes, no NaNs, `decode(encode(x))` correct range, minimal 8–16 synthetic-image overfit improves recon loss.
- **Patchification**: exact inversion `unpatchify(patchify(x)) == x` for random tensors; token-to-image occupancy map ordering.
- **Noising**: `tau=0 -> x0`; `tau=1 -> eps`; `v_target = eps - x0`; teacher receives lower noise (`min(t,s)`); same `eps` across student/teacher.
- **Time condition**: permuting per-token taus changes the corresponding token-conditioned activations, while using uniform taus yields correct shape.
- **EMA**: teacher frozen; parameter update matches expected weighted average; only update on optimizer step.
- **Loss**: finite values and gradients, `gamma=0` variant has no teacher compute, student/teacher alignment shapes correct, no silent detach of student features.
- **Sampler**: dummy constant velocity field has an analytically checkable negative-time Euler update; same fixed seed yields identical output in eval mode.
- **Train**: 2–5 optimizer steps on a synthetic dataset for all four variants, checkpoint/save/resume, `sample` CLI produces valid PNGs. OOM should be reported with advice, not swallowed.

## 9. Configuration suggestions (for exploration, NOT validated optimal values)

```yaml
seed: 42
data:
  resolution: [64, 256]
  iam_root: null                # must be configured by user
  split_dir: data/splits
  batch_size: 2
  num_workers: 0               # safe Windows default
  max_text_length: 32
model:
  ae_downsample: 4
  ae_latent_channels: 8
  patch_size: 2
  hidden_dim: 256
  depth: 8
  num_heads: 4
  student_feature_layer: 2
  teacher_feature_layer: 6
flow:
  variant: self_flow           # vanilla_fm | dual_fm | self_flow | ink_self_flow
  train_timestep_distribution: uniform
  mask_ratio: 0.25
  ink_alpha: 0.3
  ink_beta: 0.7
  rep_gamma: 0.8
  rep_warmup_steps: 2000
  ema_decay: 0.9999
  ode_solver: euler
  ode_steps: 30
train:
  max_steps: 20000            # adjustable, NOT promised convergence
  grad_accum_steps: 4
  lr: 0.0001
  weight_decay: 0.01
  gradient_clip_norm: 1.0
  amp: auto
  activation_checkpointing: true
  save_every_steps: 1000
  sample_every_steps: 500
```

The literal `data/splits` root can be expanded to train/val/test path settings by code. `rep_warmup_steps` is an engineering stabilization proposal, not from the original paper's reported default.

## 10. Required CLI examples (make them actually work; adjust argument names consistently)

```bash
# Users obtain IAM separately, per IAM license/registration.
python -m self_flow_hwg.cli.prepare_iam --iam-root /path/to/IAM --out data/splits --seed 42
python -m self_flow_hwg.cli.check_data --config configs/self_flow_tiny.yaml --output outputs/data_check

# Train AE once. All four comparison models use this SAME frozen AE checkpoint.
python -m self_flow_hwg.train.train_ae --config configs/ae_tiny.yaml --output outputs/ae

# Four ablation variants (each uses SAME AE, splits, evaluation manifest).
python -m self_flow_hwg.train.train_flow --config configs/fm_tiny.yaml --ae-checkpoint outputs/ae/best.pt --output outputs/A_fm
python -m self_flow_hwg.train.train_flow --config configs/dual_fm_tiny.yaml --ae-checkpoint outputs/ae/best.pt --output outputs/B_dual
python -m self_flow_hwg.train.train_flow --config configs/self_flow_tiny.yaml --ae-checkpoint outputs/ae/best.pt --output outputs/C_self
python -m self_flow_hwg.train.train_flow --config configs/ink_self_flow_tiny.yaml --ae-checkpoint outputs/ae/best.pt --output outputs/D_ink

# Word-level inference.
python -m self_flow_hwg.cli.sample --checkpoint outputs/C_self/best.pt --ae-checkpoint outputs/ae/best.pt --text 'minimum' --reference /path/to/ref.png --seed 17 --steps 30 --output outputs/samples/minimum.png

# Evaluation (report unavailable external metrics if weights missing).
python -m self_flow_hwg.cli.evaluate --config configs/self_flow_tiny.yaml --checkpoint outputs/C_self/best.pt --ae-checkpoint outputs/ae/best.pt --output outputs/C_self/eval
pytest -q
```

For Windows PowerShell, document equivalent commands with path quoting. Missing data should cause helpful errors; `pytest -q` and synthetic smoke CLI should work without IAM.

## 11. Implementation order / checkpoints

### P0 — first PR/commit

- Project scaffolding, configs, IAM loader/metadata, writer split, ref pairing, preprocessing, continuous AE and AE tests.
- Stop and demonstrate AE recon before training flow.

### P1 — second PR/commit

- Content/style encoders, DiT, per-token t conditioning, baseline FM, backward ODE sampler, CLI sampling, synth smoke tests.
- Show generated image grids, seed reproducibility and 1-batch overfit.

### P2 — third PR/commit

- Dual timestep, EMA teacher, two-layer-feature alignment, 4-scenario variant switch, test loss and EMA behavior.
- Verify `self_flow` differs from `dual_fm` *only* by self-distillation/EMA teacher path and loss.

### P3 — fourth PR/commit

- Ink occupancy aggregation, weighted mask sampling, comparisons and visualizations; test that target foreground is never injected as inference condition.

### P4 — fifth PR/commit

- Evaluation wrappers, per-run artifacts, stable checkpoints/resume, README with reproducible commands, A–D comparison table template, benchmark log, known limitations.

## 12. Final delivery criteria for Codex

The project is complete as **code delivery** when:

- All four variants instantiate, complete a synthetic short train loop without NaNs, save/resume and generate PNG samples with the same inference CLI.
- IAM path validation, writer-disjoint splits and same-writer different-reference pairing are tested and documented.
- Autoencoder reconstruction, latent shape and ODE sign are checked explicitly.
- No VQ-related classes are used.
- Tests run; failures and unavailable dataset-dependent checks are listed transparently.
- README describes installation, IAM preparation (user-supplied data), training, inference, evaluation, experiment comparison, and one configuration for low-memory GPU.
- Documentation cites original Self-Flow separately from the **proposed ink-aware extension**; never claim ink-aware Self-Flow already exists in the original paper or that gains are established before experiments.

## 13. Research questions for later (not MVP requirements)

- Does representation loss improve CER/WER beyond dual-timestep alone, given the same training compute?
- Does ink-aware scheduling outperform uniform random masks without increasing train-inference mismatch?
- Can reference-consistent EMA feature matching improve unseen-writer generalization without degrading diversity?
- Do character-aware/connection-aware schedules add anything beyond simple ink occupancy?
- Does the flow generator respond to reference writer when fixed text/noise is held constant?

**Important:** Implement tests and ablations first. No novelty claim or SOTA claim should be made without empirical comparison.
