#!/bin/bash
# track4_sdmonly_vast_gpuqueue.sh QUEUE_FILE [NGPU] - run ON a multi-GPU rented box after setup says READY.
# QUEUE_FILE has one run per line: "name trainer arm seed tokens train_name [flags...]"; '#' lines are comments.
# One worker per GPU (CUDA_VISIBLE_DEVICES=i); worker i takes lines i, i+N, i+2N, ... in order. A run whose
# result json exists is skipped, so the script can be started again. Threads are the box's CPU quota divided by
# the number of workers (track4_sdmonly_vast_arm.sh reads SDMONLY_WORKERS).
set -u
Q=$1; N=${2:-$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)}
R=/root/settle; L=$R/logs; A=$R/sdmllm/track4_sdmonly_vast_arm.sh
while [ ! -f "$L/READY" ]; do sleep 10; done
export SDMONLY_WORKERS=$N
worker() { i=$1; n=0
  grep -v '^#' "$Q" | grep -v '^ *$' | while read -r name trainer arm seed tok train rest; do
    if [ $(( n % N )) -eq "$i" ] && [ ! -f "$R/runs/$name.result.json" ]; then
      echo "$(date -u +%FT%TZ) gpu$i start $name" >> "$L/gpuqueue.log"
      CUDA_VISIBLE_DEVICES=$i bash "$A" "$trainer" "$name" "$arm" "$seed" "$tok" "$train" $rest > "$L/$name.out" 2>&1; rc=$?
      echo "$(date -u +%FT%TZ) gpu$i end $name rc $rc" >> "$L/gpuqueue.log"
    fi
    n=$(( n + 1 ))
  done; }
for i in $(seq 0 $(( N - 1 ))); do worker "$i" & sleep 20; done
wait
echo "$(date -u +%FT%TZ) QUEUE_DONE $Q" >> "$L/gpuqueue.log"
