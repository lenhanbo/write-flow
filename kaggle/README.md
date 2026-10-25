# Kaggle execution

Do not initialize Git from this repository until you are ready. Once it is on
GitHub, use the following cells in a Kaggle notebook.

## Inputs

Attach only the original IAM dataset to the notebook:

1. `iam-handwriting`: IAM root containing `ascii/` and `words/`.
2. Optionally, an existing run checkpoint dataset when resuming.

The default pixel-FM Kaggle config uses learned character embeddings. It does
not require WriteViT's preprocessed dataset, `unifont.pickle`, or pretrained
weights.

Kaggle mounts inputs read-only below `/kaggle/input`. Write every generated
artifact below `/kaggle/working`.

## Notebook cells

```python
!git clone https://github.com/USER/REPOSITORY.git /kaggle/working/flow-matching
%cd /kaggle/working/flow-matching
!python -m pip install -e .
```

Validate the mounted paths before consuming GPU time:

```python
from pathlib import Path

required = [
    "/kaggle/input/iam-handwriting/IAM/ascii/words.txt",
    "/kaggle/input/iam-handwriting/IAM/ascii/forms.txt",
    "/kaggle/input/iam-handwriting/IAM/words",
]
for item in required:
    assert Path(item).exists(), item
```

Create portable writer-disjoint manifests directly from the raw IAM files:

```python
!python -m self_flow_hwg.cli.prepare_iam \
  --iam-root /kaggle/input/iam-handwriting/IAM \
  --out /kaggle/working/splits \
  --seed 42 \
  --validate-images
```

The standard config then initializes the Writer encoder, learned query
embedding, query/style CrossBlocks, and Flow backbone from scratch. All model
parameters receive gradients.

Training an AE is only needed for later latent Self-Flow experiments, not for the original pixel-space FM baseline:

```python
!python -m self_flow_hwg.train.train_ae \
  --config configs/ae_kaggle.yaml \
  --output /kaggle/working/outputs/ae
```

Train the WriteViT-conditioned pixel-space vanilla Flow baseline first:

```python
!python -m self_flow_hwg.train.train_pixel_flow \
  --config configs/writevit_fm_kaggle.yaml \
  --output /kaggle/working/outputs/writevit_pixel_fm
```

Only after this baseline is validated should `configs/writevit_self_flow_kaggle.yaml`
be used for the dual-timestep + EMA teacher experiment.

To resume, attach the previous output as a Kaggle Dataset and add:

```text
--resume /kaggle/input/PREVIOUS-RUN/last.pt
```

Before ending a session, download `/kaggle/working/outputs` or publish it as a
new private Kaggle Dataset version. Files in `/kaggle/working` are not a durable
checkpoint store.

