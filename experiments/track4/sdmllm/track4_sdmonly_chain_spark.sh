#!/bin/bash
# track4_sdmonly_chain_spark.sh RUN INIT [chain flags...]
# One link of the SDMONLY chain (SDM BASE -> SDM CHAT -> WEIRD LITTLE GUY) on the Spark, resumable: it runs
# track4_sdmonly_chain_finetune.py with the Spark shard layout and the chunked loss (not compiled).
#   RUN   name of this fine-tune (checkpoints in $SDMLLM_CKPT/RUN/round_N/, results $SDMLLM_RUNS/RUN.round_N.result.json)
#   INIT  a checkpoint last.pt or a run folder (its last.pt), e.g. ~/settle24/ck/sdmonly/<base run>
# Shards: names resolve under $SDMLLM_DATA (train, train_big, val, chat_mix, chat_train, chat_test); the guy's
# shards are registered here as guy / guy_test (v1, masked prep) and guy2 / guy2_test (GUYCORPUS v2, when present).
# Layout as track4_sdmonly_spark_arm.sh. Every path can be moved by the environment (a dev test must, see below):
#   CHAIN_CODE   folder holding the python files     (default ~/settle24/sdmllm)
#   SDMLLM_CKPT  checkpoints                          (default ~/settle24/ck/sdmonly)
#   SDMLLM_RUNS  logs and result json                 (default ~/settle24/runs/sdmonly)
#   GUY_PREP     the guy's v1 prepared shards         (default ~/settle24/weirdguy/prep)
#   GUY2_DIR     the guy's v2 shards                  (default ~/settle24/sdmonly_dev/guycorpus)
# Examples:
#   track4_sdmonly_chain_spark.sh chat1 ~/settle24/ck/sdmonly/base1 --mix chat_mix:1.0 --tokens 200000000 \
#       --lr 1e-3 --extra-val val --extra-val chat_test
#   track4_sdmonly_chain_spark.sh guy1 ~/settle24/ck/sdmonly/chat1 --rounds 3 --mix guy:0.5,chat_mix:0.25,train_big:0.25 \
#       --tokens 4000000 --lr 5e-4 --extra-val guy_test --extra-val val --extra-val chat_test
# NEXT -> the table of a finished link: track4_sdmonly_chain_spark.sh --table RUN
export SDMLLM_DATA=$HOME/settle24/sdmllm/data
export SDMLLM_CKPT=${SDMLLM_CKPT:-$HOME/settle24/ck/sdmonly} SDMLLM_RUNS=${SDMLLM_RUNS:-$HOME/settle24/runs/sdmonly}
CODE=${CHAIN_CODE:-$HOME/settle24/sdmllm}
GUY_PREP=${GUY_PREP:-$HOME/settle24/weirdguy/prep}
GUY2_DIR=${GUY2_DIR:-$HOME/settle24/sdmonly_dev/guycorpus}
PY="$HOME/settle24/venv/bin/python"
cd "$CODE" || exit 1
if [ "$1" = "--table" ]; then exec "$PY" track4_sdmonly_chain_finetune.py --table "$2"; fi
[ $# -ge 2 ] || { sed -n '2,22p' "$0"; exit 2; }
RUN=$1; INIT=$2; shift 2
mkdir -p "$SDMLLM_CKPT" "$SDMLLM_RUNS"
SHARDS=()
[ -f "$GUY_PREP/train.u32" ] && SHARDS+=(--shard "guy=$GUY_PREP/train.u32" --shard "guy_test=$GUY_PREP/test.u32")
[ -f "$GUY2_DIR/guy2_train.u32" ] && SHARDS+=(--shard "guy2=$GUY2_DIR/guy2_train.u32" --shard "guy2_test=$GUY2_DIR/guy2_test.u32")
exec "$PY" track4_sdmonly_chain_finetune.py --init "$INIT" --run "$RUN" --loss chunked "${SHARDS[@]}" "$@"
