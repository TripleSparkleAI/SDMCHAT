#!/bin/bash
# THE FULL SHAPE SWEEP, WAVE SWE (sealed 2026-10-09 in THE LOG before it fires): diary 64 is the smallest diary tried
# and leads 121 at both 50M (by 0.00313) and 100M (by 0.00742), so the diary line still sits on its small edge. Rule (a)
# pushes two values past it: 36 slots a head (n_sub 6) and 16 (n_sub 4). Same shape (d768, 12 layers), recipe, data
# and seed as wave SWC (50M tokens), read against diary 64 at 50M (1.40020).
export SDMLLM_DATA=$HOME/sdmonly_base/data32k SDMLLM_CKPT=$HOME/sdmonly_base/launch/ck SDMLLM_RUNS=$HOME/sdmonly_base/launch/runs
export SDM_EVAL_TOKENS=4096 OMP_NUM_THREADS=8
P=$HOME/settle24/venv/bin/python; L=$HOME/sdmonly_base/launch/logs
cd $HOME/sdmonly_base/code
REC="--T 2048 --B 64 --accum 8 --lr 3e-3 --muon-lr 0.02 --sched wsd --compile-body --loss chunked --tokens 50000000 --train-name train_big_p200m --seed 0"
say() { echo "$(date -u +%FT%TZ) $*"; }
arm() {
  if [ -f $SDMLLM_RUNS/$1.result.json ]; then say "skip $1 (result exists)"; return; fi
  say "start $1"; $P track4_sdmonly_train.py --run $1 --arm onesdm_allsdm $REC $2 > $L/$1.out 2>&1; say "end $1 exit $?"
}
arm swe_diary36_d768_L12_50M "--d 768 --layers 12 --mem-n-sub 6"
arm swe_diary16_d768_L12_50M "--d 768 --layers 12 --mem-n-sub 4"
say "SWEEP6 DONE"
