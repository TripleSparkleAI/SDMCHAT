#!/bin/bash
# track4_sdmonly_wave7.sh QUEUE_FILE - ON a rented box: wave 7 (sealed in the plan), code in /root/settle/code_w7.
# One worker per GPU; worker i takes lines i, i+N, ... A line is "name trainer arm seed tokens train [flags...]";
# trainer "new" trains in tokens, "vocab:<data dir>" trains through the vocabulary wrapper in bytes (tokens field
# is then a byte budget) and scores by document. A run with a result json is skipped.
set -u
Q=$1; R=/root/settle; L=$R/logs; export SDMONLY_CODE=$R/code_w7; A=$SDMONLY_CODE/track4_sdmonly_vast_arm.sh
N=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l); export SDMONLY_WORKERS=$N
worker() { i=$1; n=0
  grep -v '^#' "$Q" | grep -v '^ *$' | while read -r name trainer arm seed tok train rest; do
    if [ $(( n % N )) -eq "$i" ] && [ ! -f "$R/runs/$name.result.json" ]; then
      echo "$(date -u +%FT%TZ) gpu$i start $name" >> "$L/wave7.log"
      case "$trainer" in
        vocab:*) D=${trainer#vocab:}
          (cd "$SDMONLY_CODE" && CUDA_VISIBLE_DEVICES=$i SDMLLM_CKPT=$R/ck SDMLLM_RUNS=$R/runs python track4_sdmonly_train_vocab.py \
             --data "$D" --bytes "$tok" --arm "$arm" --seed "$seed" --run "$name" --loss chunked $rest > "$L/$name.out" 2>&1
           CUDA_VISIBLE_DEVICES=$i SDMLLM_CKPT=$R/ck SDMLLM_RUNS=$R/runs python track4_sdmonly_vocab_score.py --run "$name" --data "$D" >> "$L/$name.out" 2>&1) ;;
        *) CUDA_VISIBLE_DEVICES=$i bash "$A" "$trainer" "$name" "$arm" "$seed" "$tok" "$train" $rest > "$L/$name.out" 2>&1 ;;
      esac
      echo "$(date -u +%FT%TZ) gpu$i end $name rc $?" >> "$L/wave7.log"
    fi
    n=$(( n + 1 ))
  done; }
for i in $(seq 0 $(( N - 1 ))); do worker "$i" & sleep 20; done
wait
python $SDMONLY_CODE/track4_sdmonly_summary.py --runs $R/runs --ck $R/ck --prefix w7_ --against p0_A_c > $R/runs/wave7_table.txt 2>&1
echo "$(date -u +%FT%TZ) WAVE7_DONE" >> "$L/wave7.log"
