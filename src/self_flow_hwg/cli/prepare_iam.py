from __future__ import annotations

import argparse
import json
from pathlib import Path

from self_flow_hwg.data.collate import CharacterVocabulary
from self_flow_hwg.data.iam import build_writer_disjoint_splits, parse_iam_words, save_splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate IAM and create writer-disjoint manifests")
    parser.add_argument("--iam-root", required=True); parser.add_argument("--out", required=True); parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    records, counts = parse_iam_words(args.iam_root)
    splits = build_writer_disjoint_splits(records, args.seed); save_splits(splits, args.out)
    vocab = CharacterVocabulary.from_texts(item["text"] for item in splits["train"])
    Path(args.out, "vocab.json").write_text(json.dumps(vocab.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"parse": counts, "splits": {name: {"samples": len(rows), "writers": len({r['writer_id'] for r in rows})} for name, rows in splits.items()}, "vocab_size": len(vocab)}
    Path(args.out, "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8"); print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()

