#!/bin/bash
# track4_sdmonly_spark_arm.sh TRAINER RUN ARM SEED TOKENS TRAIN_NAME [trainer flags...]
# One SDMONLY arm on the Spark, resumable. TRAINER is `new` (track4_sdmonly_train.py: sdmonly, sdmonly_dense,
# sdmonly_none) or `s0` (track4_sdmllm_train_one_arm.py: the reference arms sdm, sdm_nostore, qwen).
# Checkpoints go to ~/settle24/ck/sdmonly/RUN, logs and result json to ~/settle24/runs/sdmonly/.
# TRAIN_NAME is `train` (S0's 65.5M-token shard, the 20M-token protocol) or `train_big` (3.0B tokens).
# Loss path: compiled chunked cross-entropy, as every Spark run since SPARK24.
# Installed on the Spark as ~/settle24/code/sdmonly_arm.sh; this file is the tracked copy.
TRAINER=$1; RUN=$2; ARM=$3; SEED=$4; TOK=$5; TRAIN=$6; shift 6
export SDMLLM_DATA=$HOME/settle24/sdmllm/data SDMLLM_CKPT=$HOME/settle24/ck/sdmonly SDMLLM_RUNS=$HOME/settle24/runs/sdmonly
mkdir -p "$SDMLLM_CKPT" "$SDMLLM_RUNS"
cd "$HOME/settle24/sdmllm" || exit 1
case "$TRAINER" in
  new) PY=track4_sdmonly_train.py ;;
  s0)  PY=track4_sdmllm_train_one_arm.py ;;
  *)   echo "TRAINER must be new or s0" >&2; exit 2 ;;
esac
exec "$HOME/settle24/venv/bin/python" "$PY" --arm "$ARM" --seed "$SEED" --tokens "$TOK" --run "$RUN" \
  --train-name "$TRAIN" --loss chunked "$@"
