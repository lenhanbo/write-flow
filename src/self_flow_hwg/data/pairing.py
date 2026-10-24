from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence


def group_by_writer(records: Sequence[dict]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[str(record["writer_id"])].append(index)
    return dict(groups)


def choose_reference_index(
    index: int,
    records: Sequence[dict],
    groups: dict[str, list[int]],
    *,
    deterministic: bool,
    seed: int = 0,
) -> int:
    target = records[index]
    candidates = [
        i
        for i in groups[str(target["writer_id"])]
        if records[i]["path"] != target["path"]
    ]
    if not candidates:
        raise ValueError(f"Writer {target['writer_id']} has no different reference for {target['path']}")
    different_text = [i for i in candidates if records[i]["text"] != target["text"]]
    candidates = different_text or candidates
    rng = random.Random(f"{seed}:{index}") if deterministic else random
    return rng.choice(candidates)

