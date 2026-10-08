#!/bin/bash
# track4_sdmonly_vast_arm.sh TRAINER RUN ARM SEED TOKENS TRAIN_NAME [trainer flags...] - one arm on a rented box.
# The rented-box twin of track4_sdmonly_spark_arm.sh: /root/settle/{data,ck,runs,logs}. Threads are clamped to the
# cgroup quota (never nproc), read once from cpu.max; floor 1.
# TRAINER ddp runs track4_sdmonly_train_ddp.py (lane DDPRECIPE): pass --world N and PER-RANK --B and --accum, and set
# SDMONLY_WORKERS=N so each rank gets its share of the quota. --world 2 --B 256 --accum 32 = one GPU at --B 512 --accum 64.
TRAINER=$1; RUN=$2; ARM=$3; SEED=$4; TOK=$5; TRAIN=$6; shift 6
R=/root/settle
export SDMLLM_DATA=$R/sdmllm/data SDMLLM_CKPT=$R/ck SDMLLM_RUNS=$R/runs
Q=$(awk '{ if ($1 == "max") print 0; else printf "%d", $1 / $2 }' /sys/fs/cgroup/cpu.max 2>/dev/null); [ "${Q:-0}" -ge 1 ] || Q=4
W=${SDMONLY_WORKERS:-1}; TH=$(( Q / W )); [ "$TH" -ge 1 ] || TH=1; [ "$TH" -le 16 ] || TH=16
export OMP_NUM_THREADS=$TH MKL_NUM_THREADS=$TH OPENBLAS_NUM_THREADS=$TH NUMEXPR_NUM_THREADS=$TH
cd "${SDMONLY_CODE:-$R/sdmllm}" || exit 1
case "$TRAINER" in new) PY=track4_sdmonly_train.py ;; ddp) PY=track4_sdmonly_train_ddp.py ;; s0) PY=track4_sdmllm_train_one_arm.py ;; *) echo "TRAINER must be new, ddp or s0" >&2; exit 2 ;; esac
exec python "$PY" --arm "$ARM" --seed "$SEED" --tokens "$TOK" --run "$RUN" --train-name "$TRAIN" --loss chunked "$@"
