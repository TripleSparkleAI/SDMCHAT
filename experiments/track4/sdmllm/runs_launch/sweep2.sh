#!/bin/bash
# THE FULL SHAPE SWEEP, WAVE SWB (sealed 2026-10-08 in THE LOG before it fires): push past the two UNBRACKETED edges of
# wave SW. Depth: 24 layers at width 512 won (1.52777), so 32 and 48 go next. Diary: 1,024 slots a head beat 4,096,
# so 576 and 256 go next. Same recipe as wave SW (50M tokens, seed 0), read against the same bake-off FULL (1.54457).
export SDMLLM_DATA=$HOME/sdmonly_base/data32k SDMLLM_CKPT=$HOME/sdmonly_base/launch/ck SDMLLM_RUNS=$HOME/sdmonly_base/launch/runs
export SDM_EVAL_TOKENS=4096 OMP_NUM_THREADS=8
P=$HOME/settle24/venv/bin/python; L=$HOME/sdmonly_base/launch/logs
cd $HOME/sdmonly_base/code
REC="--T 2048 --B 64 --accum 8 --lr 3e-3 --muon-lr 0.02 --sched wsd --compile-body --loss chunked --tokens 50000000 --train-name train_big_p200m --seed 0"
say() { echo "$(date -u +%FT%TZ) $*"; }
arm() {  # run [flags]
  if [ -f $SDMLLM_RUNS/$1.result.json ]; then say "skip $1 (result exists)"; return; fi
  say "start $1"; $P track4_sdmonly_train.py --run $1 --arm onesdm_allsdm $REC $2 > $L/$1.out 2>&1; say "end $1 exit $?"
}
arm swb_diary576_d768_L12_50M  "--d 768 --layers 12 --mem-n-sub 24"
arm swb_diary256_d768_L12_50M  "--d 768 --layers 12 --mem-n-sub 16"
arm swb_deep32_d512_L32_50M    "--d 512 --layers 32"
arm swb_deep48_d512_L48_50M    "--d 512 --layers 48"
say "SWEEP2 DONE"
