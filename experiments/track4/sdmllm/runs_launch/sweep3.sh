#!/bin/bash
# THE FULL SHAPE SWEEP, WAVE SWC (sealed 2026-10-09 in THE LOG before it fires): wave SWB found the smaller diary
# winning at its smallest value (256 slots a head 1.42911, against 576 at 1.50846 and 1,024 at 1.54457), so the diary
# line is UNBRACKETED at the small edge. Push two values past it (121 and 64 slots), and test the best diary on the
# best depth so far (d512, 24 layers). Same recipe as waves SW and SWB (50M tokens, seed 0).
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
arm swc_diary121_d768_L12_50M        "--d 768 --layers 12 --mem-n-sub 11"
arm swc_diary64_d768_L12_50M         "--d 768 --layers 12 --mem-n-sub 8"
arm swc_deep24_diary256_d512_L24_50M "--d 512 --layers 24 --mem-n-sub 16"
say "SWEEP3 DONE"
