from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from self_flow_hwg.config import load_config, save_config
from self_flow_hwg.data.collate import collate_handwriting
from self_flow_hwg.flow.losses import SelfFlowObjective
from self_flow_hwg.models.autoencoder import ContinuousAutoencoder, LatentStats
from self_flow_hwg.models.dit import patchify
from self_flow_hwg.models.ema import make_ema_teacher, update_ema
from self_flow_hwg.models.system import ConditionalFlowModel
from self_flow_hwg.train.utils import SyntheticHandwritingDataset, amp_context, append_jsonl, load_prepared_dataset, make_grad_scaler, resolve_device, seed_everything


def load_ae(path: str | Path, device: torch.device) -> tuple[ContinuousAutoencoder, LatentStats]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    cfg = state.get("config", {}).get("model", {})
    model = ContinuousAutoencoder(int(cfg.get("ae_latent_channels", 8)), int(cfg.get("ae_base_channels", 32)))
    model.load_state_dict(state["model"])
    model.eval().requires_grad_(False).to(device)
    return model, LatentStats.from_state_dict(state["latent_stats"])


def train(config: dict, ae_checkpoint: str | Path, output: str | Path, *, synthetic: bool = False, resume: str | None = None) -> Path:
    output = Path(output); output.mkdir(parents=True, exist_ok=True); save_config(config, output / "config.yaml")
    seed_everything(int(config.get("seed", 42)))
    train_cfg, model_cfg, flow_cfg = config.get("train", {}), config.get("model", {}), config.get("flow", {})
    device = resolve_device(train_cfg.get("device", "auto")); ae, stats = load_ae(ae_checkpoint, device)
    if synthetic:
        vocab_size = int(model_cfg.get("vocab_size", 32)); vocabulary = {"itos": ["<pad>", "<unk>"] + [chr(32 + i) for i in range(vocab_size - 2)]}
        dataset = SyntheticHandwritingDataset(int(train_cfg.get("synthetic_samples", 16)), vocab_size=vocab_size)
    else:
        dataset, prepared_vocab = load_prepared_dataset(config, "train"); vocab_size = len(prepared_vocab); vocabulary = prepared_vocab.to_dict()
    runtime_model_cfg = {**model_cfg, "max_text_length": config.get("data", {}).get("max_text_length", 32), "activation_checkpointing": train_cfg.get("activation_checkpointing", False)}
    model = ConditionalFlowModel(vocabulary["itos"], runtime_model_cfg).to(device)
    if model.conditioning_type == "writevit":
        config.setdefault("model", {})["use_unifont"] = model.conditioning.uses_unifont
        save_config(config, output / "config.yaml")
    if model.conditioning_load_report is not None:
        report = model.conditioning_load_report
        print(f"WriteViT conditioning: loaded={len(report['loaded'])}, skipped={len(report['skipped'])}, missing={len(report['missing'])}, unexpected={len(report['unexpected'])}")
    objective = SelfFlowObjective(int(model_cfg.get("hidden_dim", 256)), int(model_cfg.get("student_feature_layer", 1)), int(model_cfg.get("teacher_feature_layer", 3))).to(device)
    variant = flow_cfg.get("variant", "self_flow")
    teacher = make_ema_teacher(model.dit) if variant in {"self_flow", "ink_self_flow"} else None
    optimizer = torch.optim.AdamW([*model.parameters(), *objective.parameters()], lr=float(train_cfg.get("lr", 1e-4)), weight_decay=float(train_cfg.get("weight_decay", .01)))
    scaler = make_grad_scaler(device, train_cfg.get("amp", "auto"))
    start = 0
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=False); model.load_state_dict(saved["model"]); objective.load_state_dict(saved["objective"]); optimizer.load_state_dict(saved["optimizer"]); start = int(saved["step"])
        if teacher is not None: teacher.load_state_dict(saved["teacher"])
        if saved.get("scaler"): scaler.load_state_dict(saved["scaler"])
    data_cfg = config.get("data", {})
    workers = int(data_cfg.get("num_workers", 0))
    loader = DataLoader(dataset, batch_size=int(data_cfg.get("batch_size", 2)), shuffle=True, collate_fn=collate_handwriting, num_workers=workers, pin_memory=device.type == "cuda", persistent_workers=workers > 0)
    iterator = iter(loader); max_steps = int(train_cfg.get("smoke_steps", 5) if synthetic else train_cfg.get("max_steps", 20000)); accumulation = int(train_cfg.get("grad_accum_steps", 1)); optimizer.zero_grad(set_to_none=True)
    for step in tqdm(range(start + 1, max_steps + 1), desc=variant):
        try: batch = next(iterator)
        except StopIteration: iterator = iter(loader); batch = next(iterator)
        images = batch["target_image"].to(device); references = batch["reference_image"].to(device); foreground = batch["foreground_mask"].to(device)
        with torch.no_grad(): clean = patchify(stats.normalize(ae.encode(images)), model.dit.patch_size)
        with amp_context(device, train_cfg.get("amp", "auto")):
            conditions = model.conditions(batch["text_ids"].to(device), batch["text_mask"].to(device), references)
            warmup = max(1, int(flow_cfg.get("rep_warmup_steps", 0))); gamma = float(flow_cfg.get("rep_gamma", .8)) * min(1.0, step / warmup)
            loss, terms = objective(variant, model.dit, teacher, clean, *conditions, foreground, float(flow_cfg.get("mask_ratio", .25)), gamma, float(flow_cfg.get("ink_alpha", .3)), float(flow_cfg.get("ink_beta", .7)))
            scaled = loss / accumulation
        scaler.scale(scaled).backward()
        if step % accumulation == 0 or step == max_steps:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([*model.parameters(), *objective.parameters()], float(train_cfg.get("gradient_clip_norm", 1)))
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            if teacher is not None: update_ema(teacher, model.dit, float(flow_cfg.get("ema_decay", .9999)))
        append_jsonl(output / "metrics.jsonl", {"step": step, **{k: float(v) for k, v in terms.items()}})
        save_every = int(train_cfg.get("save_every_steps", 1000))
        if save_every > 0 and step % save_every == 0 and step < max_steps:
            torch.save({"model": model.state_dict(), "objective": objective.state_dict(), "teacher": teacher.state_dict() if teacher else None, "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "config": config, "step": step, "vocab": vocabulary, "rng_state": torch.get_rng_state()}, output / "last.pt")
    checkpoint = output / "last.pt"
    payload = {"model": model.state_dict(), "objective": objective.state_dict(), "teacher": teacher.state_dict() if teacher else None, "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "config": config, "step": max_steps, "vocab": vocabulary, "rng_state": torch.get_rng_state()}
    torch.save(payload, checkpoint); torch.save(payload, output / "best.pt")
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--ae-checkpoint", required=True); parser.add_argument("--output", required=True); parser.add_argument("--resume"); parser.add_argument("--synthetic-smoke", action="store_true"); args = parser.parse_args()
    try:
        print(train(load_config(args.config), args.ae_checkpoint, args.output, synthetic=args.synthetic_smoke, resume=args.resume))
    except RuntimeError as error:
        if "out of memory" in str(error).lower(): raise RuntimeError("CUDA OOM: reduce data.batch_size/model width/depth/style tokens, increase grad accumulation, or enable activation checkpointing.") from error
        raise


if __name__ == "__main__": main()
