from __future__ import annotations

from collections.abc import Iterable

import torch


class CharacterVocabulary:
    def __init__(self, characters: Iterable[str]):
        unique = sorted(set(characters))
        self.itos = ["<pad>", "<unk>", *unique]
        self.stoi = {char: index for index, char in enumerate(self.itos)}

    @classmethod
    def from_texts(cls, texts: Iterable[str]) -> "CharacterVocabulary":
        return cls(char for text in texts for char in text)

    def encode(self, text: str, max_length: int | None = None) -> torch.Tensor:
        if max_length is not None and len(text) > max_length:
            raise ValueError(f"Text length {len(text)} exceeds max_length={max_length}: {text!r}")
        return torch.tensor([self.stoi.get(char, 1) for char in text], dtype=torch.long)

    def to_dict(self) -> dict:
        return {"itos": self.itos}

    @classmethod
    def from_dict(cls, data: dict) -> "CharacterVocabulary":
        vocab = cls([])
        vocab.itos = list(data["itos"])
        vocab.stoi = {char: index for index, char in enumerate(vocab.itos)}
        return vocab

    def __len__(self) -> int:
        return len(self.itos)


def collate_handwriting(batch: list[dict]) -> dict:
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    lengths = [int(item["text_ids"].numel()) for item in batch]
    max_length = max(lengths)
    ids = torch.zeros(len(batch), max_length, dtype=torch.long)
    mask = torch.zeros(len(batch), max_length, dtype=torch.bool)
    for row, item in enumerate(batch):
        length = lengths[row]
        ids[row, :length] = item["text_ids"]
        mask[row, :length] = True
    result = {
        "target_image": torch.stack([x["target_image"] for x in batch]),
        "reference_image": torch.stack([x["reference_image"] for x in batch]),
        "foreground_mask": torch.stack([x["foreground_mask"] for x in batch]),
        "text_ids": ids,
        "text_mask": mask,
        "text": [x["text"] for x in batch],
        "writer_id": [x["writer_id"] for x in batch],
        "target_path": [x["target_path"] for x in batch],
        "reference_path": [x["reference_path"] for x in batch],
    }
    return result

