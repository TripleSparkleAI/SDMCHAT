#!/bin/bash
# SDMLLM S0 run queue: runs "arm:seed" pairs in order, each resumable; a finished run (result json present) is skipped.
# usage: bash track4_sdmllm_run_queue.sh <queue-name> arm:seed [arm:seed ...]
cd "$(dirname "$0")"
Q=$1; shift
TOK=${SDMLLM_TOKENS:-20000000}
for pair in "$@"; do
  arm=${pair%%:*}; seed=${pair##*:}
  run="${arm}_s${seed}_$((TOK/1000000))M"
  if [ -f "runs/${run}.result.json" ]; then echo "skip $run"; continue; fi
  echo "$(date -u +%FT%TZ) start $run" >> "runs/queue_${Q}.log"
  python3 track4_sdmllm_train_one_arm.py --arm "$arm" --seed "$seed" --tokens "$TOK" --run "$run" > "runs/${run}.stdout" 2>&1
  echo "$(date -u +%FT%TZ) end $run exit $?" >> "runs/queue_${Q}.log"
done
