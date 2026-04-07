#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/3d/teacher
PY=$ROOT/.venv/bin/python
CONFIG=$ROOT/configs/paper_re10k_vggt_certaintyband_final.yaml
PRECOMP_LOG=$ROOT/outputs/paper_re10k_final_precompute.log
TRAIN_LOG=$ROOT/outputs/paper_re10k_final_train.log
EVAL_LOG=$ROOT/outputs/paper_re10k_final_eval.log
DUAL_LOG=$ROOT/outputs/paper_re10k_final_dualthreshold.log
PRECOMP_SHARDS=${TEACHER_PRECOMP_SHARDS:-4}
PRECOMP_DEVICE=${TEACHER_PRECOMP_DEVICE:-cuda}
PRECOMP_SHARD_LOG_DIR=$ROOT/outputs/paper_re10k_final_precompute_shards

mkdir -p "$PRECOMP_SHARD_LOG_DIR"
: > "$PRECOMP_LOG"

echo "[precompute] starting sharded VGGT packet generation with ${PRECOMP_SHARDS} shards on ${PRECOMP_DEVICE}" | tee -a "$PRECOMP_LOG"
pids=()
for (( shard=0; shard<PRECOMP_SHARDS; shard++ )); do
  shard_log="$PRECOMP_SHARD_LOG_DIR/shard_${shard}.log"
  echo "[precompute] launch shard ${shard}/${PRECOMP_SHARDS} -> ${shard_log}" | tee -a "$PRECOMP_LOG"
  $PY $ROOT/scripts/precompute_vggt_packets.py \
    --config "$CONFIG" \
    --limit -1 \
    --device "$PRECOMP_DEVICE" \
    --shard-index "$shard" \
    --num-shards "$PRECOMP_SHARDS" \
    > "$shard_log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

echo "[precompute] all shards finished" | tee -a "$PRECOMP_LOG"
$PY $ROOT/scripts/train.py --config $CONFIG > $TRAIN_LOG 2>&1
$PY $ROOT/scripts/eval.py --config $CONFIG --output $ROOT/outputs/local_ablation/full_paper_re10k_vggt_certaintyband_final.json > $EVAL_LOG 2>&1
$PY $ROOT/scripts/dual_threshold_sweep.py --config $CONFIG --output $ROOT/outputs/local_ablation/dual_threshold_paper_re10k_vggt_certaintyband_final.json --teacher-target-root /root/3d/teacher/outputs/vggt_packets_paper_rawsupport --enable-layered-eval --min-visible-f1 0.30 --max-visible-bleed 0.20 > $DUAL_LOG 2>&1
