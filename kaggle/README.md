# Kaggle execution

Do not initialize Git from this repository until you are ready. Once it is on
GitHub, use the following cells in a Kaggle notebook.

## Inputs

Attach these datasets to the notebook:

1. `iam-handwriting`: IAM root containing `ascii/` and `words/`.
2. `iam-splits`: portable manifests produced by `prepare_iam` (paths inside the
   manifests are relative to the IAM root).
3. `writevit-assets`: `unifont.pickle` from WriteViT.
4. `writevit-checkpoint`: the WriteViT checkpoint to reuse.
5. Optionally, an existing AE/checkpoint dataset when resuming.

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
    "/kaggle/input/iam-splits/splits/train.json",
    "/kaggle/input/writevit-assets/unifont.pickle",
    "/kaggle/input/writevit-checkpoint/model.pth",
]
for item in required:
    assert Path(item).exists(), item
```

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

