from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from self_flow_hwg.config import load_config, save_config
from self_flow_hwg.data.collate import collate_handwriting
from self_flow_hwg.models.autoencoder import ContinuousAutoencoder, reconstruction_loss
from self_flow_hwg.eval.grids import save_contact_sheet
from self_flow_hwg.train.utils import SyntheticHandwritingDataset, amp_context, append_jsonl, load_prepared_dataset, make_grad_scaler, resolve_device, seed_everything


def train(config: dict, output: str | Path, *, synthetic: bool = False, resume: str | None = None) -> Path:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    save_config(config, output / "config.yaml")
    seed_everything(int(config.get("seed", 42)))
    device = resolve_device(config.get("train", {}).get("device", "auto"))
    model_cfg, train_cfg = config.get("model", {}), config.get("train", {})
    model = ContinuousAutoencoder(int(model_cfg.get("ae_latent_channels", 8)), int(model_cfg.get("ae_base_channels", 32))).to(device)
    dataset = SyntheticHandwritingDataset(int(train_cfg.get("synthetic_samples", 16))) if synthetic else load_prepared_dataset(config, "train")[0]
    data_cfg = config.get("data", {})
    workers = int(data_cfg.get("num_workers", 0))
    loader = DataLoader(dataset, batch_size=int(data_cfg.get("batch_size", 2)), shuffle=True, collate_fn=collate_handwriting, num_workers=workers, pin_memory=device.type == "cuda", persistent_workers=workers > 0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.get("lr", 2e-4)))
    scaler = make_grad_scaler(device, train_cfg.get("amp", "auto"))
    start = 0
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=False); model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"]); start = int(saved["step"])
        if saved.get("scaler"): scaler.load_state_dict(saved["scaler"])
    steps = int(train_cfg.get("smoke_steps", 5) if synthetic else train_cfg.get("max_steps", 10000))
    model.train()
    iterator = iter(loader)
    for step in tqdm(range(start + 1, steps + 1), desc="AE"):
        try: batch = next(iterator)
        except StopIteration: iterator = iter(loader); batch = next(iterator)
        image, foreground = batch["target_image"].to(device), batch["foreground_mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with amp_context(device, train_cfg.get("amp", "auto")):
            prediction = model(image)
            loss, terms = reconstruction_loss(prediction, image, foreground, float(train_cfg.get("ink_weight", 2)), float(train_cfg.get("edge_weight", .2)))
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("gradient_clip_norm", 1)))
        scaler.step(optimizer); scaler.update()
        append_jsonl(output / "metrics.jsonl", {"step": step, "loss": float(loss.detach()), **{k: float(v) for k, v in terms.items()}})
        save_every = int(train_cfg.get("save_every_steps", 1000))
        if save_every > 0 and step % save_every == 0 and step < steps:
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "config": config, "step": step, "rng_state": torch.get_rng_state()}, output / "last.pt")
    model.eval()
    latents = []
    with torch.no_grad():
        for batch in loader:
            latents.append(model.encode(batch["target_image"].to(device)).cpu())
    stacked = torch.cat(latents)
    stats = {"mean": stacked.mean((0, 2, 3), keepdim=True), "std": stacked.std((0, 2, 3), keepdim=True).clamp_min(1e-6)}
    with torch.no_grad():
        preview = next(iter(loader))["target_image"].to(device)
        reconstruction = model(preview)
        save_contact_sheet(torch.cat((preview.cpu(), reconstruction.cpu())), output / "recon_grid.png", nrow=len(preview))
    checkpoint = output / "last.pt"
    payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "latent_stats": stats, "config": config, "step": steps, "rng_state": torch.get_rng_state()}
    torch.save(payload, checkpoint); torch.save(payload, output / "best.pt")
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--synthetic-smoke", action="store_true"); parser.add_argument("--resume")
    args = parser.parse_args()
    try:
        print(train(load_config(args.config), args.output, synthetic=args.synthetic_smoke, resume=args.resume))
    except RuntimeError as error:
        if "out of memory" in str(error).lower(): raise RuntimeError("CUDA OOM: reduce data.batch_size/model width, increase grad accumulation, or enable activation checkpointing.") from error
        raise


if __name__ == "__main__": main()

