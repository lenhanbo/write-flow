from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.transforms.functional import to_pil_image

from self_flow_hwg.data.preprocess import load_word_image
from self_flow_hwg.flow.sampler import ode_sample
from self_flow_hwg.models.dit import unpatchify
from self_flow_hwg.models.system import ConditionalFlowModel
from self_flow_hwg.train.train_flow import load_ae
from self_flow_hwg.train.utils import resolve_device


@torch.no_grad()
def generate(checkpoint: str, ae_checkpoint: str, text: str, reference: str, output: str, seed: int = 17, steps: int = 30, solver: str = "euler", device_name: str = "auto") -> Path:
    device = resolve_device(device_name); saved = torch.load(checkpoint, map_location="cpu", weights_only=False); cfg = saved["config"]; model_cfg = cfg["model"]
    vocab = saved["vocab"]["itos"]; stoi = {char: i for i, char in enumerate(vocab)}
    if len(text) > int(cfg["data"].get("max_text_length", 32)): raise ValueError("Query text exceeds configured max_text_length")
    runtime_model_cfg = {**model_cfg, "max_text_length": cfg["data"].get("max_text_length", 32), "activation_checkpointing": False, "writevit_checkpoint": None, "unifont_path": None}
    model = ConditionalFlowModel(vocab, runtime_model_cfg).to(device); model.load_state_dict(saved["model"])
    if saved.get("teacher") is not None: model.dit.load_state_dict(saved["teacher"])
    model.eval(); ae, stats = load_ae(ae_checkpoint, device)
    ref, _ = load_word_image(reference, size=tuple(cfg["data"].get("resolution", [64, 256]))); ref = ref.unsqueeze(0).to(device)
    ids = torch.tensor([[stoi.get(char, 1) for char in text]], device=device); mask = torch.ones_like(ids, dtype=torch.bool); conditions = model.conditions(ids, mask, ref)
    generator = torch.Generator(device=device).manual_seed(seed); raw_dim = model.dit.latent_channels * model.dit.patch_size**2
    noise = torch.randn((1, model.dit.num_tokens, raw_dim), generator=generator, device=device)
    def velocity(x, tau): return model.dit(x, tau, *conditions)[0]
    tokens = ode_sample(velocity, noise, steps, solver); latent = unpatchify(tokens, model.dit.latent_channels, *model.dit.latent_size, model.dit.patch_size)
    image = ae.decode(stats.denormalize(latent)).clamp(0, 1)[0].cpu(); output_path = Path(output); output_path.parent.mkdir(parents=True, exist_ok=True); to_pil_image(image).save(output_path); return output_path


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--checkpoint", required=True); p.add_argument("--ae-checkpoint", required=True); p.add_argument("--text", required=True); p.add_argument("--reference", required=True); p.add_argument("--output", required=True); p.add_argument("--seed", type=int, default=17); p.add_argument("--steps", type=int, default=30); p.add_argument("--solver", choices=["euler", "heun"], default="euler"); p.add_argument("--device", default="auto"); a = p.parse_args(); print(generate(a.checkpoint, a.ae_checkpoint, a.text, a.reference, a.output, a.seed, a.steps, a.solver, a.device))


if __name__ == "__main__": main()

