from __future__ import annotations


def edit_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1): current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def cer(predictions: list[str], targets: list[str]) -> float:
    return sum(edit_distance(p, t) for p, t in zip(predictions, targets, strict=True)) / max(1, sum(map(len, targets)))

