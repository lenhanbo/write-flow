from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image

from self_flow_hwg.config import load_config
from self_flow_hwg.data.collate import CharacterVocabulary
from self_flow_hwg.data.iam import IAMWordDataset


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--output", required=True); parser.add_argument("--limit", type=int, default=16); args = parser.parse_args()
    cfg = load_config(args.config); split_dir = Path(cfg["data"]["split_dir"]); manifest = split_dir / "train.json"
    if not manifest.is_file(): raise FileNotFoundError(f"Prepared split missing: {manifest}. Run prepare_iam first.")
    records = json.loads(manifest.read_text(encoding="utf-8"))["records"]
    vocab = CharacterVocabulary.from_dict(json.loads((split_dir / "vocab.json").read_text(encoding="utf-8")))
    ds = IAMWordDataset(records, vocab, split="val", image_size=tuple(cfg["data"].get("resolution", [64, 256])), iam_root=cfg["data"].get("iam_root"))
    rows = []
    for i in range(min(args.limit, len(ds))):
        item = ds[i]; row = Image.new("L", (512, 84), 255); row.paste(to_pil_image(item["reference_image"]), (0, 20)); row.paste(to_pil_image(item["target_image"]), (256, 20)); ImageDraw.Draw(row).text((2, 2), f"ref | target={item['text']}", fill=0); rows.append(row)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    sheet = Image.new("L", (512, 84 * len(rows)), 255)
    for i, row in enumerate(rows): sheet.paste(row, (0, i * 84))
    sheet.save(output / "data_check.png"); print(f"Checked {len(rows)} pairs -> {output / 'data_check.png'}")


if __name__ == "__main__": main()

