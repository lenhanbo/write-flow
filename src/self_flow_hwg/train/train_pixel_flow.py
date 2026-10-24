from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from self_flow_hwg.config import load_config, save_config
from self_flow_hwg.data.collate import collate_handwriting
from self_flow_hwg.flow.pixel import pixel_ot_batch
from self_flow_hwg.models.pixel_flow import WriteViTPixelFlow
from self_flow_hwg.train.utils import (
    SyntheticHandwritingDataset,
    amp_context,
    append_jsonl,
    load_prepared_dataset,
    make_grad_scaler,
    resolve_device,
    seed_everything,
)


def _payload(model, optimizer, scaler, config, vocabulary, step):
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "config": config,
        "vocab": vocabulary,
        "step": step,
        "rng_state": torch.get_rng_state(),
        "space": "pixel",
        "flow_convention": "noise_t0_data_t1",
    }


def train(config: dict, output: str | Path, *, synthetic: bool = False, resume: str | None = None) -> Path:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(int(config.get("seed", 42)))
    data_cfg, model_cfg, flow_cfg, train_cfg = (
        config.get("data", {}), config.get("model", {}), config.get("flow", {}), config.get("train", {})
    )
    device = resolve_device(train_cfg.get("device", "auto"))
    if synthetic:
        vocab_size = int(model_cfg.get("vocab_size", 32))
        vocabulary = {"itos": ["<pad>", "<unk>"] + [chr(32 + index) for index in range(vocab_size - 2)]}
        dataset = SyntheticHandwritingDataset(int(train_cfg.get("synthetic_samples", 16)), vocab_size=vocab_size)
    else:
        dataset, prepared_vocab = load_prepared_dataset(config, "train")
        vocabulary = prepared_vocab.to_dict()
    runtime_cfg = {**model_cfg, "max_text_length": data_cfg.get("max_text_length", 32)}
    model = WriteViTPixelFlow(vocabulary["itos"], runtime_cfg).to(device)
    config.setdefault("model", {})["use_unifont"] = model.conditioning.uses_unifont
    save_config(config, output / "config.yaml")
    if model.conditioning_load_report is not None:
        report = model.conditioning_load_report
        print(f"WriteViT conditioning: loaded={len(report['loaded'])}, skipped={len(report['skipped'])}, missing={len(report['missing'])}")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=float(train_cfg.get("lr", 1e-4)), weight_decay=float(train_cfg.get("weight_decay", .01)))
    scaler = make_grad_scaler(device, train_cfg.get("amp", "auto"))
    start = 0
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=False)
        if saved.get("space") != "pixel":
            raise ValueError("Resume checkpoint is not a pixel-space Flow Matching checkpoint")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        if saved.get("scaler"):
            scaler.load_state_dict(saved["scaler"])
        start = int(saved["step"])
    workers = int(data_cfg.get("num_workers", 0))
    loader = DataLoader(
        dataset, batch_size=int(data_cfg.get("batch_size", 2)), shuffle=True,
        collate_fn=collate_handwriting, num_workers=workers, pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    iterator = iter(loader)
    max_steps = int(train_cfg.get("smoke_steps", 5) if synthetic else train_cfg.get("max_steps", 20000))
    accumulation = int(train_cfg.get("grad_accum_steps", 1))
    sigma_min = float(flow_cfg.get("sigma_min", .001))
    optimizer.zero_grad(set_to_none=True)
    for step in tqdm(range(start + 1, max_steps + 1), desc="pixel_fm"):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        # Pixel-space FM uses symmetric image data; no autoencoder is constructed or called.
        data = batch["target_image"].to(device).mul(2).sub(1)
        reference = batch["reference_image"].to(device)
        flow_batch = pixel_ot_batch(data, sigma_min)
        with amp_context(device, train_cfg.get("amp", "auto")):
            prediction = model(
                flow_batch.state, flow_batch.timestep, batch["text_ids"].to(device),
                batch["text_mask"].to(device), reference,
            )
            loss = F.mse_loss(prediction, flow_batch.velocity)
            scaled_loss = loss / accumulation
        scaler.scale(scaled_loss).backward()
        optimizer_step = step % accumulation == 0 or step == max_steps
        if optimizer_step:
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, float(train_cfg.get("gradient_clip_norm", 1)))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        else:
            gradient_norm = torch.zeros((), device=device)
        append_jsonl(output / "metrics.jsonl", {"step": step, "loss": float(loss.detach()), "gradient_norm": float(gradient_norm)})
        save_every = int(train_cfg.get("save_every_steps", 1000))
        if save_every > 0 and step % save_every == 0 and step < max_steps:
            torch.save(_payload(model, optimizer, scaler, config, vocabulary, step), output / "last.pt")
    payload = _payload(model, optimizer, scaler, config, vocabulary, max_steps)
    torch.save(payload, output / "last.pt")
    torch.save(payload, output / "best.pt")
    return output / "last.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train WriteViT-conditioned pixel-space Flow Matching")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--synthetic-smoke", action="store_true")
    args = parser.parse_args()
    try:
        print(train(load_config(args.config), args.output, synthetic=args.synthetic_smoke, resume=args.resume))
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            raise RuntimeError("CUDA OOM: reduce batch/base_channels, increase accumulation, or reduce style tokens.") from error
        raise


if __name__ == "__main__":
    main()

