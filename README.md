# Teacher3D

`Teacher3D` is a research codebase for `VGGT teacher -> single-view student` scene reconstruction.
The current paper-line method is `certaintyband_localplus`: visible certainty + boundary-local hidden certainty + deep-hidden ambiguity.

## Environment

```bash
source /root/3d/teacher/.venv/bin/activate
pip install -e .
```

## Dependency

Prepare the upstream VGGT code under `third_party/vggt` before running training or packet precompute.

## Final Paper Config

Main config:

```text
configs/paper_re10k_vggt_certaintyband_final.yaml
```

One-shot full pipeline:

```bash
bash scripts/run_paper_re10k_final.sh
```

This pipeline runs:

1. full RE10K VGGT packet precompute
2. full training
3. full evaluation
4. dual-threshold evaluation

## Main Scripts

Train:

```bash
python scripts/train.py --config configs/paper_re10k_vggt_certaintyband_final.yaml
```

Eval:

```bash
python scripts/eval.py --config configs/paper_re10k_vggt_certaintyband_final.yaml
```

Dual-threshold eval:

```bash
python scripts/dual_threshold_sweep.py   --config configs/paper_re10k_vggt_certaintyband_final.yaml   --output outputs/local_ablation/dual_threshold_paper_re10k_vggt_certaintyband_final.json   --teacher-target-root /root/3d/teacher/outputs/vggt_packets_paper_rawsupport   --enable-layered-eval   --min-visible-f1 0.30   --max-visible-bleed 0.20
```

Visualizations:

```bash
python scripts/visualize.py --config configs/paper_re10k_vggt_certaintyband_final.yaml --max-samples 8
```

## Core Files

- `src/teacher3d/vggt_integration.py`: VGGT packet export
- `src/teacher3d/teacher.py`: teacher target construction
- `src/teacher3d/losses.py`: certainty / ambiguity losses
- `src/teacher3d/eval.py`: global + boundary-local metrics
- `configs/paper_re10k_vggt_certaintyband_final.yaml`: final full-dataset config
- `scripts/run_paper_re10k_final.sh`: one-shot paper pipeline

## Notes

- `outputs/` is ignored by git.
- Historical ablation configs are kept under `configs/` for reference.
- Current paper-line notes are in `paper_line_2026-04-07_certaintyband.md`.
