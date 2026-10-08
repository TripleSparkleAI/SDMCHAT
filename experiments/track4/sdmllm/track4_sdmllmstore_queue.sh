#!/bin/bash
# SDMLLMSTORE run queue. Each spec file line is "run_name arm seed [trainer flags...]"; lines run in order, each
# resumable, a run with a result json is skipped, '#' lines are comments. The spec file is re-read after every run,
# so lines appended later (the sealed conditional sweep) are picked up. Every run is 20M tokens, the S0 protocol.
# usage: bash track4_sdmllmstore_queue.sh <spec-file> <queue-name>
cd "$(dirname "$0")"
SPEC=$1; Q=$2
export SDMLLM_DATA=${SDMLLM_DATA:-/Users/happyrobot/Code/TripleSparkle/_flows/dwarfstar/experiments/track4/sdmllm/data}
while true; do
  line=$(grep -v '^#' "$SPEC" | grep -v '^ *$' | while read -r run arm seed rest; do
    [ -f "runs/${run}.result.json" ] && continue
    [ -f "runs_store/claim_${run}" ] && [ "$(cat runs_store/claim_${run})" != "$Q" ] && continue
    echo "$run $arm $seed $rest"; break; done)
  [ -z "$line" ] && { echo "$(date -u +%FT%TZ) queue $Q empty" >> "runs_store/queue_${Q}.log"; exit 0; }
  set -- $line; run=$1; arm=$2; seed=$3; shift 3
  echo "$Q" > "runs_store/claim_${run}"
  echo "$(date -u +%FT%TZ) start $run load $(sysctl -n vm.loadavg) $(pmset -g | grep powermode)" >> "runs_store/queue_${Q}.log"
  python3 track4_sdmllm_train_one_arm.py --arm "$arm" --seed "$seed" --tokens 20000000 --run "$run" "$@" > "runs/${run}.stdout" 2>&1
  echo "$(date -u +%FT%TZ) end $run exit $? load $(sysctl -n vm.loadavg)" >> "runs_store/queue_${Q}.log"
  [ -f "runs/${run}.result.json" ] || { echo "$(date -u +%FT%TZ) $run FAILED, queue stops" >> "runs_store/queue_${Q}.log"; exit 1; }
done
