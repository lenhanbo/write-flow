import torch

from self_flow_hwg.data.collate import CharacterVocabulary, collate_handwriting
from self_flow_hwg.data.iam import build_writer_disjoint_splits
from self_flow_hwg.data.pairing import choose_reference_index, group_by_writer
from self_flow_hwg.data.preprocess import compute_foreground


def test_writer_disjoint_and_reference_pairing():
    records = [{"writer_id": str(w), "path": f"{w}-{i}", "text": f"word{i}"} for w in range(10) for i in range(2)]
    splits = build_writer_disjoint_splits(records, seed=1)
    writer_sets = [{r["writer_id"] for r in split} for split in splits.values()]
    assert not (writer_sets[0] & writer_sets[1] or writer_sets[0] & writer_sets[2] or writer_sets[1] & writer_sets[2])
    groups = group_by_writer(records)
    ref = choose_reference_index(0, records, groups, deterministic=True, seed=3)
    assert records[ref]["path"] != records[0]["path"]


def test_vocab_collate_and_foreground():
    vocab = CharacterVocabulary.from_texts(["Ab!", "A?"])
    assert vocab.encode("Z").item() == 1
    image = torch.ones(1, 64, 256); image[:, 10:20, 20:30] = 0
    item = {"target_image": image, "reference_image": image, "foreground_mask": compute_foreground(image), "text_ids": vocab.encode("Ab"), "text": "Ab", "writer_id": "w", "target_path": "a", "reference_path": "b"}
    batch = collate_handwriting([item, {**item, "text_ids": vocab.encode("A")}])
    assert batch["text_ids"].shape == (2, 2)
    assert batch["text_mask"].tolist() == [[True, True], [True, False]]
    assert batch["foreground_mask"].max() == 1

