from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.transforms.functional import to_pil_image

from self_flow_hwg.data.preprocess import load_word_image
from self_flow_hwg.flow.pixel import pixel_ode_sample
from self_flow_hwg.models.pixel_flow import WriteViTPixelFlow
from self_flow_hwg.train.utils import resolve_device


@torch.no_grad()
def generate(
    checkpoint: str, text: str, reference: str, output: str, seed: int = 17,
    steps: int = 30, solver: str = "euler", device_name: str = "auto",
) -> Path:
    device = resolve_device(device_name)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved.get("space") != "pixel":
        raise ValueError("Checkpoint is not a pixel-space Flow Matching checkpoint")
    config = saved["config"]
    vocabulary = saved["vocab"]["itos"]
    stoi = {character: index for index, character in enumerate(vocabulary)}
    model_cfg = {
        **config["model"], "max_text_length": config["data"].get("max_text_length", 32),
        "writevit_checkpoint": None, "unifont_path": None,
    }
    model = WriteViTPixelFlow(vocabulary, model_cfg).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    image_size = tuple(config["data"].get("resolution", [64, 256]))
    reference_image, _ = load_word_image(reference, size=image_size)
    reference_image = reference_image.unsqueeze(0).to(device)
    ids = torch.tensor([[stoi.get(character, 1) for character in text]], device=device)
    if ids.shape[1] > int(config["data"].get("max_text_length", 32)):
        raise ValueError("Query exceeds configured max_text_length")
    text_mask = torch.ones_like(ids, dtype=torch.bool)
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn((1, 1, *image_size), generator=generator, device=device)

    def velocity(state: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return model(state, timestep, ids, text_mask, reference_image)

    generated = pixel_ode_sample(velocity, noise, steps, solver)
    image = generated.add(1).div(2).clamp(0, 1)[0].cpu()
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    to_pil_image(image).save(output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample WriteViT-conditioned pixel-space Flow Matching")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--solver", choices=["euler", "heun"], default="euler")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    print(generate(args.checkpoint, args.text, args.reference, args.output, args.seed, args.steps, args.solver, args.device))


if __name__ == "__main__":
    main()
