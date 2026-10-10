#!/bin/bash
# THE NEW FULL BASE (sealed 2026-10-10 in THE LOG before it fires): the winning shape of waves SW to SWE, FULL (arm
# onesdm_allsdm: run-time SDM diary + trained SDM encyclopedia, no attention, no MLP), d768, 12 layers, diary 36 slots
# a head (n_sub 6), 2.0B tokens of train_big (one pass, 3.1B in the file), WSD, the shape-sweep recipe, seed 0,
# with a kept checkpoint at 1.1B for the comparison with the old FULL base.
export SDMLLM_DATA=$HOME/sdmonly_base/data32k SDMLLM_CKPT=$HOME/sdmonly_base/launch/ck SDMLLM_RUNS=$HOME/sdmonly_base/launch/runs
export SDM_EVAL_TOKENS=4096 OMP_NUM_THREADS=8
P=$HOME/settle24/venv/bin/python; L=$HOME/sdmonly_base/launch/logs
cd $HOME/sdmonly_base/code
R=base2_fullsdm_d768_L12_diary36_T2048_2000M
say() { echo "$(date -u +%FT%TZ) $*"; }
if [ -f $SDMLLM_RUNS/$R.result.json ]; then say "skip $R (result exists)"; exit 0; fi
say "start $R"
$P track4_sdmonly_train.py --run $R --arm onesdm_allsdm --T 2048 --B 64 --accum 8 --lr 3e-3 --muon-lr 0.02 --sched wsd \
  --compile-body --loss chunked --tokens 2000000000 --train-name train_big --seed 0 \
  --d 768 --layers 12 --mem-n-sub 6 --keep-at 1100000000 >> $L/$R.out 2>&1
say "end $R exit $?"
