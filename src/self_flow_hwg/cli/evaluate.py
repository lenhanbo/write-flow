from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--config", required=True); p.add_argument("--checkpoint", required=True); p.add_argument("--ae-checkpoint", required=True); p.add_argument("--output", required=True); p.add_argument("--recognizer-checkpoint"); p.add_argument("--style-checkpoint"); a = p.parse_args()
    output = Path(a.output); output.mkdir(parents=True, exist_ok=True)
    result = {"cer": "unavailable" if not a.recognizer_checkpoint else "not_implemented", "wer": "unavailable" if not a.recognizer_checkpoint else "not_implemented", "style_similarity": "unavailable" if not a.style_checkpoint else "not_implemented", "note": "External recognizer/style weights are required; no metrics are fabricated."}
    (output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8"); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()

