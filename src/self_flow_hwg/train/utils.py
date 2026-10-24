from __future__ import annotations

import json
import random
from contextlib import nullcontext
from pathlib import Path

import torch

from self_flow_hwg.data.collate import CharacterVocabulary
from self_flow_hwg.data.iam import IAMWordDataset


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def amp_context(device: torch.device, mode: str = "auto"):
    if device.type != "cuda" or mode == "off":
        return nullcontext()
    dtype = torch.bfloat16 if mode in {"auto", "bf16"} and torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def make_grad_scaler(device: torch.device, mode: str = "auto") -> torch.amp.GradScaler:
    fp16 = device.type == "cuda" and (mode == "fp16" or (mode == "auto" and not torch.cuda.is_bf16_supported()))
    return torch.amp.GradScaler("cuda", enabled=fp16)


def append_jsonl(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


class SyntheticHandwritingDataset(torch.utils.data.Dataset):
    """Dependency-free stroke-like data used only for tests and smoke runs."""
    def __init__(self, size: int = 16, image_size: tuple[int, int] = (64, 256), vocab_size: int = 32, seed: int = 0):
        generator = torch.Generator().manual_seed(seed)
        self.items = []
        for index in range(size):
            image = torch.ones(1, *image_size)
            y = 12 + index % 24
            for offset in range(4):
                x0 = 8 + offset * 35 + index % 7
                image[:, y : y + 3, x0 : x0 + 24] = torch.rand(1, generator=generator) * 0.15
                image[:, y - 5 : y + 10, x0 + offset * 2 : x0 + offset * 2 + 2] = 0.0
            reference = torch.roll(image, shifts=(index % 5) - 2, dims=-2)
            ids = torch.tensor([2 + index % (vocab_size - 2), 2 + (index + 1) % (vocab_size - 2)])
            self.items.append({"target_image": image, "reference_image": reference, "foreground_mask": 1 - image, "text_ids": ids, "text": "ab", "writer_id": str(index // 2), "target_path": f"synthetic:{index}", "reference_path": f"synthetic-ref:{index}"})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def load_prepared_dataset(config: dict, split: str) -> tuple[IAMWordDataset, CharacterVocabulary]:
    data_cfg = config.get("data", {})
    split_dir = Path(data_cfg.get("split_dir", "data/splits"))
    manifest_path, vocab_path = split_dir / f"{split}.json", split_dir / "vocab.json"
    missing = [str(path) for path in (manifest_path, vocab_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Prepared IAM files are missing: " + ", ".join(missing) + ". Run python -m self_flow_hwg.cli.prepare_iam first.")
    records = json.loads(manifest_path.read_text(encoding="utf-8"))["records"]
    vocabulary = CharacterVocabulary.from_dict(json.loads(vocab_path.read_text(encoding="utf-8")))
    dataset = IAMWordDataset(
        records, vocabulary, split=split, image_size=tuple(data_cfg.get("resolution", [64, 256])),
        margin=int(data_cfg.get("margin", 4)), foreground_threshold=float(data_cfg.get("foreground_threshold", .8)),
        seed=int(config.get("seed", 42)), max_text_length=int(data_cfg.get("max_text_length", 32)),
        iam_root=data_cfg.get("iam_root"),
    )
    return dataset, vocabulary

