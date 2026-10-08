#!/bin/bash
# track4_sdmonly_fill.sh FILL_QUEUE - ON a rented box: give each GPU that has been idle for two checks in a row
# (60 s apart, so the box's own wave runner has clearly finished with it) the next unclaimed line of FILL_QUEUE.
# A line is "name trainer arm seed tokens train [flags...]", as in track4_sdmonly_wave7.sh. A line is claimed by
# mkdir (atomic), skipped if its result json exists, and run through track4_sdmonly_vast_arm.sh. Exits when every
# line is claimed and no claimed run is still going.
set -u
Q=$1; R=/root/settle; L=$R/logs; C=$R/fill_claims; mkdir -p "$C"
export SDMONLY_CODE=$R/code_w7; A=$SDMONLY_CODE/track4_sdmonly_vast_arm.sh
N=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l); export SDMONLY_WORKERS=$N
busy() {  # 0 if a python training process runs on GPU $1
  for p in $(ps -eo pid,args | awk '/[p]ython track4_sdmonly_train|[p]ython track4_sdmllm_train/ {print $1}'); do
    d=$(tr '\0' '\n' < /proc/$p/environ 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p')
    [ "$d" = "$1" ] && return 0
  done; return 1; }
declare -A idle
while true; do
  left=0
  for g in $(seq 0 $((N - 1))); do
    if busy "$g"; then idle[$g]=0; continue; fi
    idle[$g]=$(( ${idle[$g]:-0} + 1 ))
    [ "${idle[$g]}" -ge 2 ] || continue
    while read -r name trainer arm seed tok train rest; do
      [ -n "$name" ] || continue
      [ -f "$R/runs/$name.result.json" ] && continue
      mkdir "$C/$name" 2>/dev/null || continue
      echo "$(date -u +%FT%TZ) gpu$g start $name (fill)" >> "$L/wave7.log"
      (CUDA_VISIBLE_DEVICES=$g nohup bash "$A" "$trainer" "$name" "$arm" "$seed" "$tok" "$train" $rest > "$L/$name.out" 2>&1 < /dev/null; echo "$(date -u +%FT%TZ) gpu$g end $name (fill) rc $?" >> "$L/wave7.log") &
      idle[$g]=0; break
    done < <(grep -v '^#' "$Q" | grep -v '^ *$')
  done
  for n in $(grep -v '^#' "$Q" | awk 'NF {print $1}'); do [ -d "$C/$n" ] || left=1; done
  [ "$left" -eq 0 ] && ! ps -eo args | grep -q "[p]ython track4_sdm" && { echo "$(date -u +%FT%TZ) FILL_DONE $Q" >> "$L/wave7.log"; exit 0; }
  sleep 60
done
