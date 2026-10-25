from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from torch.utils.data import Dataset

from .collate import CharacterVocabulary
from .pairing import choose_reference_index, group_by_writer
from .preprocess import compute_foreground, load_word_image


def parse_iam_words(iam_root: str | Path) -> tuple[list[dict], dict[str, int]]:
    """Parse IAM words.txt and forms.txt; never infer writers from word filenames."""
    root = Path(iam_root)
    words_file = root / "ascii" / "words.txt"
    forms_file = root / "ascii" / "forms.txt"
    missing = [str(path) for path in (words_file, forms_file, root / "words") if not path.exists()]
    if missing:
        raise FileNotFoundError("IAM root is incomplete. Missing: " + ", ".join(missing))
    form_writers: dict[str, str] = {}
    for raw in forms_file.read_text(encoding="utf-8", errors="replace").splitlines():
        if not raw or raw.startswith("#"):
            continue
        fields = raw.split()
        if len(fields) >= 2:
            form_writers[fields[0]] = fields[1]
    records: list[dict] = []
    counts = {"ok": 0, "segmentation_error": 0, "missing_image": 0, "missing_writer": 0}
    for raw in words_file.read_text(encoding="utf-8", errors="replace").splitlines():
        if not raw or raw.startswith("#"):
            continue
        fields = raw.split(maxsplit=8)
        if len(fields) < 9:
            continue
        word_id, status, text = fields[0], fields[1], fields[8]
        if status != "ok":
            counts["segmentation_error"] += 1
            continue
        form_id = "-".join(word_id.split("-")[:2])
        writer_id = form_writers.get(form_id)
        if writer_id is None:
            counts["missing_writer"] += 1
            continue
        first, second = word_id.split("-")[:2]
        path = root / "words" / first / f"{first}-{second}" / f"{word_id}.png"
        if not path.is_file():
            counts["missing_image"] += 1
            continue
        records.append({"word_id": word_id, "form_id": form_id, "writer_id": writer_id, "text": text, "path": path.relative_to(root).as_posix()})
        counts["ok"] += 1
    return records, counts


def parse_gt_manifest(iam_root: str | Path, filename: str, *, validate_images: bool = False) -> tuple[list[dict], dict[str, int]]:
    """Parse the ``relative/path.png<TAB>text`` IAM package used in ../data.

    This package omits IAM's forms-to-writer metadata. We conservatively treat
    each form as its own style group: words from one form are guaranteed to have
    one writer, while words from different forms are never assumed to match.
    """
    root = Path(iam_root).resolve()
    manifest = root / filename
    if not manifest.is_file():
        raise FileNotFoundError(f"Ground-truth manifest not found: {manifest}")
    records: list[dict] = []
    counts = {"ok": 0, "malformed": 0, "missing_image": 0, "images_validated": int(validate_images)}
    for line_number, raw in enumerate(manifest.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if "\t" not in raw:
            counts["malformed"] += 1
            continue
        relative, text = raw.split("\t", 1)
        relative_path = Path(*relative.replace("\\", "/").split("/"))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Manifest path escapes IAM root at {filename}:{line_number}: {relative}")
        path = root / relative_path
        if validate_images and not path.is_file():
            counts["missing_image"] += 1
            continue
        parts = relative_path.parts
        if len(parts) < 3:
            counts["malformed"] += 1
            continue
        form_id = parts[-2]
        word_id = path.stem
        records.append({
            "word_id": word_id,
            "form_id": form_id,
            "writer_id": f"form:{form_id}",
            "writer_id_source": "form_proxy",
            "text": text,
            "path": path.relative_to(root).as_posix(),
        })
        counts["ok"] += 1
    return records, counts


def build_packaged_gt_splits(iam_root: str | Path, seed: int = 42) -> tuple[dict[str, list[dict]], dict]:
    """Use packaged train/val lists, then split validation groups into val/test."""
    train, train_counts = parse_gt_manifest(iam_root, "train_gt.txt")
    held_out, val_counts = parse_gt_manifest(iam_root, "val_gt.txt")
    train_groups = {row["writer_id"] for row in train}
    overlap_groups = train_groups & {row["writer_id"] for row in held_out}
    held_out = [row for row in held_out if row["writer_id"] not in overlap_groups]
    group_counts: dict[str, int] = {}
    for row in [*train, *held_out]:
        group_counts[row["writer_id"]] = group_counts.get(row["writer_id"], 0) + 1
    train = [row for row in train if group_counts[row["writer_id"]] >= 2]
    held_out = [row for row in held_out if group_counts[row["writer_id"]] >= 2]
    groups = sorted({row["writer_id"] for row in held_out})
    random.Random(seed).shuffle(groups)
    midpoint = len(groups) // 2
    val_groups = set(groups[:midpoint])
    splits = {
        "train": train,
        "val": [row for row in held_out if row["writer_id"] in val_groups],
        "test": [row for row in held_out if row["writer_id"] not in val_groups],
    }
    summary = {
        "source": "packaged_gt",
        "writer_id_source": "form_proxy",
        "warning": "forms.txt is absent; each form is conservatively treated as a separate style group",
        "train_manifest": train_counts,
        "val_manifest": val_counts,
        "excluded_overlapping_form_groups": len(overlap_groups),
    }
    return splits, summary


def build_writer_disjoint_splits(
    records: list[dict], seed: int = 42, ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
) -> dict[str, list[dict]]:
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1, got {ratios}")
    groups = group_by_writer(records)
    eligible = [writer for writer, indexes in groups.items() if len(indexes) >= 2]
    random.Random(seed).shuffle(eligible)
    n = len(eligible)
    boundaries = (round(n * ratios[0]), round(n * (ratios[0] + ratios[1])))
    writer_splits = {
        "train": set(eligible[: boundaries[0]]),
        "val": set(eligible[boundaries[0] : boundaries[1]]),
        "test": set(eligible[boundaries[1] :]),
    }
    return {name: [r for r in records if r["writer_id"] in writers] for name, writers in writer_splits.items()}


def save_splits(splits: dict[str, list[dict]], output: str | Path) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, records in splits.items():
        payload = {"records": records, "sha256": hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()}
        (output / f"{name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


class IAMWordDataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        vocabulary: CharacterVocabulary,
        *,
        split: str = "train",
        image_size: tuple[int, int] = (64, 256),
        margin: int = 4,
        foreground_threshold: float = 0.8,
        seed: int = 42,
        max_text_length: int = 32,
        iam_root: str | Path | None = None,
    ):
        self.records = records
        self.vocabulary = vocabulary
        self.split = split
        self.image_size = image_size
        self.margin = margin
        self.foreground_threshold = foreground_threshold
        self.seed = seed
        self.max_text_length = max_text_length
        self.iam_root = Path(iam_root) if iam_root is not None else None
        self.groups = group_by_writer(records)
        bad = [writer for writer, indices in self.groups.items() if len(indices) < 2]
        if bad:
            raise ValueError(f"Writers need at least two valid words for reference pairing: {bad[:5]}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        target = self.records[index]
        ref_index = choose_reference_index(index, self.records, self.groups, deterministic=self.split != "train", seed=self.seed)
        reference = self.records[ref_index]
        target_path = Path(target["path"])
        reference_path = Path(reference["path"])
        if not target_path.is_absolute():
            if self.iam_root is None:
                raise ValueError("Manifest uses relative image paths but IAMWordDataset.iam_root was not provided")
            target_path = self.iam_root / target_path
        if not reference_path.is_absolute():
            if self.iam_root is None:
                raise ValueError("Manifest uses relative image paths but IAMWordDataset.iam_root was not provided")
            reference_path = self.iam_root / reference_path
        target_image, _ = load_word_image(target_path, size=self.image_size, margin=self.margin)
        reference_image, _ = load_word_image(reference_path, size=self.image_size, margin=self.margin)
        return {
            "target_image": target_image,
            "reference_image": reference_image,
            "text_ids": self.vocabulary.encode(target["text"], self.max_text_length),
            "text": target["text"],
            "writer_id": str(target["writer_id"]),
            "target_path": str(target_path),
            "reference_path": str(reference_path),
            "foreground_mask": compute_foreground(target_image, self.foreground_threshold),
        }

