from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch

from teacher3d.config import load_config
from teacher3d.data import build_dataloader
from teacher3d.losses import LossComputer
from teacher3d.models import Teacher3DV1
from teacher3d.teacher import build_teacher


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch(batch, device: torch.device, non_blocking: bool = False):
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=non_blocking)
        else:
            moved[key] = value
    return moved


def resolve_amp_dtype(device: torch.device, requested: str) -> torch.dtype | None:
    if device.type != "cuda":
        return None
    name = str(requested).lower()
    if name == "bfloat16":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if name == "float16":
        return torch.float16
    raise ValueError(f"Unsupported amp_dtype: {requested}")


def train_one_epoch(
    model,
    loader,
    teacher,
    loss_computer,
    optimizer,
    device,
    log_every,
    max_steps,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    scaler,
    non_blocking: bool,
):
    model.train()
    metrics = []
    autocast_enabled = amp_enabled and amp_dtype is not None
    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        start_time = time.perf_counter()
        batch = move_batch(batch, device, non_blocking=non_blocking)
        teacher_targets = teacher(batch)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
            outputs = model(batch["image"])
            losses = loss_computer(outputs, batch, teacher_targets)
        if scaler.is_enabled():
            scaler.scale(losses["total"]).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses["total"].backward()
            optimizer.step()
        step_time = time.perf_counter() - start_time
        step_metrics = {name: float(value.detach().cpu()) for name, value in losses.items()}
        step_metrics["step_time"] = step_time
        step_metrics["samples_per_sec"] = float(batch["image"].shape[0]) / max(step_time, 1e-6)
        metrics.append(step_metrics)
        if step % log_every == 0:
            print(json.dumps({"step": step, **step_metrics}, sort_keys=True))
    if not metrics:
        return {"total": 0.0}
    summary = {key: sum(item[key] for item in metrics) / max(len(metrics), 1) for key in metrics[0]}
    return summary


def main(config_path: str) -> None:
    config = load_config(config_path)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(config.seed))

    device = torch.device(config.train.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(getattr(config.train, "cudnn_benchmark", True))
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision(str(getattr(config.train, "float32_matmul_precision", "high")))
    loader = build_dataloader(config, shuffle=True)
    teacher = build_teacher(config)
    model = Teacher3DV1(config).to(device)
    loss_computer = LossComputer(config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.optim.lr),
        weight_decay=float(config.optim.weight_decay),
    )

    amp_enabled = bool(getattr(config.train, "amp", device.type == "cuda"))
    amp_dtype = resolve_amp_dtype(device, getattr(config.train, "amp_dtype", "bfloat16")) if amp_enabled else None
    scaler_enabled = bool(amp_enabled and amp_dtype == torch.float16 and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=scaler_enabled)
    non_blocking = bool(getattr(config.train, "non_blocking", device.type == "cuda"))
    max_steps_value = int(getattr(config.train, "max_steps_per_epoch", -1))
    max_steps = None if max_steps_value <= 0 else max_steps_value

    history = []
    for epoch in range(int(config.train.epochs)):
        summary = train_one_epoch(
            model=model,
            loader=loader,
            teacher=teacher,
            loss_computer=loss_computer,
            optimizer=optimizer,
            device=device,
            log_every=int(config.train.log_every),
            max_steps=max_steps,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            scaler=scaler,
            non_blocking=non_blocking,
        )
        summary["epoch"] = epoch
        history.append(summary)
        print(json.dumps({"epoch_summary": summary}, sort_keys=True))

    torch.save(model.state_dict(), output_dir / "model.pt")
    with (output_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    print(f"saved model to {output_dir / 'model.pt'}")
