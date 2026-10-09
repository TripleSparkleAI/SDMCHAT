#!/bin/bash
# THE YARDSTICK AT 100M (sealed 2026-10-09 in THE LOG before it fires): wave SWD trains FULL at 100M tokens, and the
# only transformer at this shape and data is at 50M (bk_yard, 1.22265). Same shape (d768, 12 layers), same data and
# recipe as wave SWD, seed 0: the fair 100M comparison. Arm yardstick_transformer (a transformer, trained only to
# measure against).
export SDMLLM_DATA=$HOME/sdmonly_base/data32k SDMLLM_CKPT=$HOME/sdmonly_base/launch/ck SDMLLM_RUNS=$HOME/sdmonly_base/launch/runs
export SDM_EVAL_TOKENS=4096 OMP_NUM_THREADS=8
P=$HOME/settle24/venv/bin/python; L=$HOME/sdmonly_base/launch/logs
cd $HOME/sdmonly_base/code
REC="--T 2048 --B 64 --accum 8 --lr 3e-3 --muon-lr 0.02 --sched wsd --compile-body --loss chunked --tokens 100000000 --train-name train_big_p200m --seed 0"
R=swd_yard_d768_L12_T2048_100M
say() { echo "$(date -u +%FT%TZ) $*"; }
if [ -f $SDMLLM_RUNS/$R.result.json ]; then say "skip $R (result exists)"; exit 0; fi
say "start $R"; $P track4_sdmonly_train.py --run $R --arm yardstick_transformer $REC --d 768 --layers 12 > $L/$R.out 2>&1; say "end $R exit $?"
say "SWEEP5 DONE"
