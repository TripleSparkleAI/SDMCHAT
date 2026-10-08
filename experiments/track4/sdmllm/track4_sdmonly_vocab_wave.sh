#!/bin/bash
# track4_sdmonly_vocab_wave.sh - THE TOKENIZER WAVE, run ON a rented box from /root/settle/sdmllm after setup says READY.
#
# The same model (sdmonly centre arm: d 256, 4 hops, n_sub 256, seed 0, the wave-1 protocol) trained on the SAME TEXT
# with four tokenizers, at EQUAL TRAINING BYTES (the bytes of 20M DeepSeek tokens), scored in bits per byte on the same
# held-out documents (val docs 1,000 to 1,999, by document). One GPU, one run at a time.
#   arm        tokenizer                                         shard set
#   vw_v129k   DeepSeek V4, 129,280, as is (the bridge to wave 1)  data_v129k -> links to data/
#   vw_v65k    trained byte-level BPE 65,536 (FineWeb-Edu)         data_v65k
#   vw_v32k    trained byte-level BPE 32,768 (the recommendation)  data_v32k
#   vw_v16k    trained byte-level BPE 16,384                       data_v16k
# Stages: SHARDS (decode the box's own data/train.u32 and val.u32 back to text, re-encode three ways; ~2 min on an M5)
# -> BYTES (20M x the DeepSeek set's measured bytes per token) -> TRAIN x4 -> SCORE x4 -> TABLE.
# Idempotent: a stage with a marker in /root/settle/logs/, a run whose result json carries "vocab_wave", and a run whose
# result json carries "vocab_score" are skipped, so the script can be started again after a cut. A cut run resumes from
# its last.pt (the trainer's own resume). Needs in /root/settle/sdmllm/: the three vocab python files, tokenizers/*.json,
# and the code the S0 and SDM-only trainers already need. Logs: /root/settle/logs/vw_*.log. Table:
# /root/settle/runs/vocab_wave_table.txt.
# Env: DRY=1 prints the commands and runs nothing. VW_REF_TOKENS (default 20000000). VW_ARMS (default "v129k v65k v32k v16k").
#      VW_ROOT (default /root/settle; an M5 test points it at a scratch copy). VW_LOSS (default chunked; plain on MPS).
#      VW_EXTRA (extra trainer flags, appended; for a smoke only).
# Start (the coordinator, on the box):
#   cd /root/settle/sdmllm && nohup bash track4_sdmonly_vocab_wave.sh > /root/settle/logs/vocab_wave.out 2>&1 &
set -u
R=${VW_ROOT:-/root/settle}; L=$R/logs; C=$R/sdmllm
mkdir -p "$L" "$R/ck" "$R/runs"
cd "$C" || exit 1
export SDMLLM_CKPT=$R/ck SDMLLM_RUNS=$R/runs
Q=$(awk '{ if ($1 == "max") print 0; else printf "%d", $1 / $2 }' /sys/fs/cgroup/cpu.max 2>/dev/null); [ "${Q:-0}" -ge 1 ] || Q=4
TH=$Q; [ "$TH" -le 16 ] || TH=16
export OMP_NUM_THREADS=$TH MKL_NUM_THREADS=$TH OPENBLAS_NUM_THREADS=$TH NUMEXPR_NUM_THREADS=$TH
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$L/vocab_wave.log"; }
doit() { if [ "${DRY:-0}" = 1 ]; then echo "DRY: $*"; else "$@"; fi; }
ARMS=${VW_ARMS:-"v129k v65k v32k v16k"}
REF=${VW_REF_TOKENS:-20000000}

[ -f "$L/READY" ] || say "WARNING: no $L/READY marker (setup not finished?); continuing"
for f in track4_sdmonly_vocab_shards.py track4_sdmonly_train_vocab.py track4_sdmonly_vocab_score.py \
         tokenizers/trained_bpe_65536.json tokenizers/trained_bpe_32768.json tokenizers/trained_bpe_16384.json \
         data/train.u32 data/val.u32 data/token_bytes.i32 data/ds4_v4flash_tokenizer.json; do
  [ -f "$f" ] || { say "MISSING $C/$f"; exit 1; }
done

if [ ! -f "$L/VW_SHARDS" ]; then
  say "SHARDS start"
  doit python track4_sdmonly_vocab_shards.py --src data --out-root "$C" --deepseek-as-is v129k \
      --tok v65k=tokenizers/trained_bpe_65536.json --tok v32k=tokenizers/trained_bpe_32768.json \
      --tok v16k=tokenizers/trained_bpe_16384.json > "$L/vw_shards.log" 2>&1 \
    || { say "SHARDS FAILED (see $L/vw_shards.log)"; exit 1; }
  [ "${DRY:-0}" = 1 ] || { touch "$L/VW_SHARDS"; say "SHARDS done: $(grep -c '"event": "built"' "$L/vw_shards.log") sets built"; }
fi

if [ "${DRY:-0}" = 1 ] && [ ! -f data_v129k/meta.json ]; then BYTES=95417201; else
BYTES=$(python -c "import json; m=json.load(open('data_v129k/meta.json')); print(int(round($REF*m['splits']['train']['bytes_per_token'])))") \
  || { say "BYTES FAILED"; exit 1; }; fi
say "BYTES $BYTES = $REF DeepSeek tokens x the data_v129k train bytes per token"

has() { [ -f "$R/runs/$1.result.json" ] && grep -q "\"$2\"" "$R/runs/$1.result.json"; }
G="--eval-every 500 --ckpt-every 500 --log-every 100"
for v in $ARMS; do
  run=vw_$v
  if has "$run" vocab_wave; then say "skip train $run"; else
    say "TRAIN start $run"
    doit python track4_sdmonly_train_vocab.py --data "$C/data_$v" --bytes "$BYTES" --arm sdmonly --d 256 --hops 4 \
        --n-sub 256 --seed 0 --run "$run" --loss "${VW_LOSS:-chunked}" $G ${VW_EXTRA:-} >> "$L/vw_$v.log" 2>&1
    say "TRAIN end $run rc $?"
  fi
  if has "$run" vocab_score; then say "skip score $run"; else
    doit python track4_sdmonly_vocab_score.py --run "$run" --data "$C/data_$v" >> "$L/vw_$v.score.log" 2>&1
    say "SCORE $run rc $? $(grep VOCAB_SCORE "$L/vw_$v.score.log" 2>/dev/null | tail -1 | cut -c1-200)"
  fi
done

LIST=$(for v in $ARMS; do printf "vw_%s," "$v"; done); LIST=${LIST%,}
doit python track4_sdmonly_vocab_score.py --table "$LIST" > "$R/runs/vocab_wave_table.txt.tmp" 2>&1 \
  && { [ "${DRY:-0}" = 1 ] || mv "$R/runs/vocab_wave_table.txt.tmp" "$R/runs/vocab_wave_table.txt"; }
[ "${DRY:-0}" = 1 ] || cat "$R/runs/vocab_wave_table.txt" | tee -a "$L/vocab_wave.log"
say "VOCAB_WAVE_DONE"
echo "NEXT -> scp the table and runs/vw_*.result.json home; python3 track4_sdmonly_summary.py"
