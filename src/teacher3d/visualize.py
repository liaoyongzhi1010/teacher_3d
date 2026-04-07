from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from teacher3d.calibration import apply_eval_calibration
from teacher3d.config import load_config
from teacher3d.data import build_dataloader
from teacher3d.models import Teacher3DV1
from teacher3d.train import move_batch


def _to_uint8_rgb(image: torch.Tensor) -> Image.Image:
    array = image.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return Image.fromarray((array * 255.0).astype(np.uint8))


def _normalize_map(tensor: torch.Tensor, value_min: float | None = None, value_max: float | None = None) -> np.ndarray:
    array = tensor.detach().cpu().float().numpy()
    if value_min is None:
        value_min = float(np.nanmin(array))
    if value_max is None:
        value_max = float(np.nanmax(array))
    scale = max(value_max - value_min, 1e-6)
    array = np.clip((array - value_min) / scale, 0.0, 1.0)
    return (array * 255.0).astype(np.uint8)


def _to_gray_image(tensor: torch.Tensor, value_min: float | None = None, value_max: float | None = None) -> Image.Image:
    return Image.fromarray(_normalize_map(tensor, value_min=value_min, value_max=value_max), mode='L').convert('RGB')


def _add_label(image: Image.Image, text: str) -> Image.Image:
    labeled = image.copy()
    draw = ImageDraw.Draw(labeled)
    draw.rectangle((0, 0, image.width, 18), fill=(0, 0, 0))
    draw.text((4, 3), text, fill=(255, 255, 255))
    return labeled


PANEL_ORDER = [
    'input',
    'target',
    'rendered',
    'visible_conf_pred',
    'visible_conf_teacher',
    'hidden_conf_pred',
    'hidden_conf_teacher',
    'visible_depth_pred',
    'visible_depth_teacher',
    'hidden_depth_pred',
    'hidden_depth_teacher',
]


def _build_panels(batch_item: dict[str, torch.Tensor], output_item: dict[str, torch.Tensor]) -> dict[str, Image.Image]:
    panels: dict[str, Image.Image] = {}
    panels['input'] = _to_uint8_rgb(batch_item['image'])
    panels['target'] = _to_uint8_rgb(batch_item['target_image'])
    panels['rendered'] = _to_uint8_rgb(output_item['rendered_target'])

    panels['visible_conf_pred'] = _to_gray_image(output_item['visible_confidence'][0], 0.0, 1.0)
    if 'teacher_visible_confidence' in batch_item:
        panels['visible_conf_teacher'] = _to_gray_image(batch_item['teacher_visible_confidence'][0], 0.0, 1.0)
    else:
        panels['visible_conf_teacher'] = Image.new('RGB', panels['input'].size, (0, 0, 0))

    panels['hidden_conf_pred'] = _to_gray_image(output_item['hidden_confidence'][0], 0.0, 1.0)
    if 'teacher_hidden_confidence' in batch_item:
        panels['hidden_conf_teacher'] = _to_gray_image(batch_item['teacher_hidden_confidence'][0], 0.0, 1.0)
    else:
        panels['hidden_conf_teacher'] = Image.new('RGB', panels['input'].size, (0, 0, 0))

    if 'teacher_visible_depth' in batch_item:
        vmin = min(float(output_item['visible_depth'].min().cpu()), float(batch_item['teacher_visible_depth'].min().cpu()))
        vmax = max(float(output_item['visible_depth'].max().cpu()), float(batch_item['teacher_visible_depth'].max().cpu()))
        panels['visible_depth_teacher'] = _to_gray_image(batch_item['teacher_visible_depth'][0], vmin, vmax)
    else:
        vmin = float(output_item['visible_depth'].min().cpu())
        vmax = float(output_item['visible_depth'].max().cpu())
        panels['visible_depth_teacher'] = Image.new('RGB', panels['input'].size, (0, 0, 0))
    panels['visible_depth_pred'] = _to_gray_image(output_item['visible_depth'][0], vmin, vmax)

    if 'teacher_hidden_depth' in batch_item:
        hmin = min(float(output_item['hidden_depth'].min().cpu()), float(batch_item['teacher_hidden_depth'].min().cpu()))
        hmax = max(float(output_item['hidden_depth'].max().cpu()), float(batch_item['teacher_hidden_depth'].max().cpu()))
        panels['hidden_depth_teacher'] = _to_gray_image(batch_item['teacher_hidden_depth'][0], hmin, hmax)
    else:
        hmin = float(output_item['hidden_depth'].min().cpu())
        hmax = float(output_item['hidden_depth'].max().cpu())
        panels['hidden_depth_teacher'] = Image.new('RGB', panels['input'].size, (0, 0, 0))
    panels['hidden_depth_pred'] = _to_gray_image(output_item['hidden_depth'][0], hmin, hmax)

    return panels


def _save_grid(panels: dict[str, Image.Image], output_path: Path) -> None:
    ordered = [_add_label(panels[name], name) for name in PANEL_ORDER]
    cols = 3
    rows = (len(ordered) + cols - 1) // cols
    width, height = ordered[0].size
    grid = Image.new('RGB', (cols * width, rows * height), (20, 20, 20))
    for idx, image in enumerate(ordered):
        x = (idx % cols) * width
        y = (idx // cols) * height
        grid.paste(image, (x, y))
    grid.save(output_path)


def _split_batch(batch, outputs, index: int):
    batch_item = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            batch_item[key] = value[index].detach().cpu()
        else:
            batch_item[key] = value[index] if isinstance(value, list) else value
    output_item = {key: value[index].detach().cpu() for key, value in outputs.items() if torch.is_tensor(value)}
    return batch_item, output_item


def export_visualizations(config_path: str, checkpoint_path: str | None = None, output_dir: str | None = None, device_override: str | None = None, max_samples: int = 4) -> None:
    config = load_config(config_path)
    checkpoint = Path(checkpoint_path) if checkpoint_path else Path(config.output_dir) / 'model.pt'
    out_dir = Path(output_dir) if output_dir else Path(config.output_dir) / 'visuals'
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_config = getattr(config, 'eval', None)
    device_name = device_override or getattr(eval_config, 'device', None) or config.train.device
    device = torch.device(device_name)

    loader = build_dataloader(config, shuffle=False)
    model = Teacher3DV1(config).to(device)
    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    saved = 0
    with torch.inference_mode():
        for batch in loader:
            batch = move_batch(batch, device)
            outputs = apply_eval_calibration(model(batch['image']), eval_config)
            batch_size = int(batch['image'].shape[0])
            for index in range(batch_size):
                if saved >= max_samples:
                    return
                batch_item, output_item = _split_batch(batch, outputs, index)
                scene_id = batch_item.get('scene_id', f'sample{saved:03d}')
                context_ts = int(batch_item['context_timestamp'].item())
                target_ts = int(batch_item['target_timestamp'].item())
                sample_dir = out_dir / f'{saved:03d}_{scene_id}_{context_ts}_{target_ts}'
                sample_dir.mkdir(parents=True, exist_ok=True)
                panels = _build_panels(batch_item, output_item)
                for name, image in panels.items():
                    image.save(sample_dir / f'{name}.png')
                _save_grid(panels, sample_dir / 'grid.png')
                saved += 1


def main(config_path: str, checkpoint_path: str | None = None, output_dir: str | None = None, device_override: str | None = None, max_samples: int = 4) -> None:
    export_visualizations(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        device_override=device_override,
        max_samples=max_samples,
    )
    target_dir = output_dir or 'default visual output dir'
    print(f'saved visualizations to {target_dir}')
