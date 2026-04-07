from __future__ import annotations

import json
import random
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


def move_batch(batch, device: torch.device):
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def train_one_epoch(model, loader, teacher, loss_computer, optimizer, device, log_every, max_steps):
    model.train()
    metrics = []
    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        batch = move_batch(batch, device)
        teacher_targets = teacher(batch)
        outputs = model(batch["image"])
        losses = loss_computer(outputs, batch, teacher_targets)
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        optimizer.step()
        step_metrics = {name: float(value.detach().cpu()) for name, value in losses.items()}
        metrics.append(step_metrics)
        if step % log_every == 0:
            print(json.dumps({"step": step, **step_metrics}, sort_keys=True))
    summary = {key: sum(item[key] for item in metrics) / max(len(metrics), 1) for key in metrics[0]}
    return summary


def main(config_path: str) -> None:
    config = load_config(config_path)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(config.seed))

    device = torch.device(config.train.device)
    loader = build_dataloader(config, shuffle=True)
    teacher = build_teacher(config)
    model = Teacher3DV1(config).to(device)
    loss_computer = LossComputer(config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.optim.lr),
        weight_decay=float(config.optim.weight_decay),
    )

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
            max_steps=int(config.train.max_steps_per_epoch),
        )
        summary["epoch"] = epoch
        history.append(summary)
        print(json.dumps({"epoch_summary": summary}, sort_keys=True))

    torch.save(model.state_dict(), output_dir / "model.pt")
    with (output_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    print(f"saved model to {output_dir / 'model.pt'}")
