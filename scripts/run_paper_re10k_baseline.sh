#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/3d/teacher
PY=$ROOT/.venv/bin/python
CONFIG=$ROOT/configs/paper_re10k_vggt_baseline_full.yaml
TRAIN_LOG=$ROOT/outputs/paper_re10k_baseline_train.log
EVAL_LOG=$ROOT/outputs/paper_re10k_baseline_eval.log
DUAL_LOG=$ROOT/outputs/paper_re10k_baseline_dualthreshold.log

$PY $ROOT/scripts/train.py --config $CONFIG > $TRAIN_LOG 2>&1
$PY $ROOT/scripts/eval.py --config $CONFIG --output $ROOT/outputs/local_ablation/full_paper_re10k_vggt_baseline_full.json > $EVAL_LOG 2>&1
$PY $ROOT/scripts/dual_threshold_sweep.py --config $CONFIG --output $ROOT/outputs/local_ablation/dual_threshold_paper_re10k_vggt_baseline_full.json --teacher-target-root /root/3d/teacher/outputs/vggt_packets_paper_rawsupport --enable-layered-eval --min-visible-f1 0.30 --max-visible-bleed 0.20 > $DUAL_LOG 2>&1
