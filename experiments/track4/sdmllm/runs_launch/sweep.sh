#!/bin/bash
# THE FULL SHAPE SWEEP (sealed 2026-10-07 in THE LOG before it fires). Seven FULL shapes at 50M tokens each, the
# bake-off's exact recipe, so each is read against bk_allsdm_d768_L12_T2048_50M (TEST 1.54457). sw_seed1 is the same
# shape on seed 1: it measures the noise floor every other arm is judged by. Each arm skips itself if its result exists.
export SDMLLM_DATA=$HOME/sdmonly_base/data32k SDMLLM_CKPT=$HOME/sdmonly_base/launch/ck SDMLLM_RUNS=$HOME/sdmonly_base/launch/runs
export SDM_EVAL_TOKENS=4096 OMP_NUM_THREADS=8
P=$HOME/settle24/venv/bin/python; L=$HOME/sdmonly_base/launch/logs
cd $HOME/sdmonly_base/code
REC="--T 2048 --B 64 --accum 8 --lr 3e-3 --muon-lr 0.02 --sched wsd --compile-body --loss chunked --tokens 50000000 --train-name train_big_p200m"
say() { echo "$(date -u +%FT%TZ) $*"; }
arm() {  # run arm [flags]
  if [ -f $SDMLLM_RUNS/$1.result.json ]; then say "skip $1 (result exists)"; return; fi
  say "start $1"; $P track4_sdmonly_train.py --run $1 --arm $2 $REC $3 > $L/$1.out 2>&1; say "end $1 exit $?"
}
arm sw_seed1_d768_L12_50M      onesdm_allsdm     "--d 768 --layers 12 --seed 1"
arm sw_slots4x_d768_L12_50M    onesdm_allsdm_big "--d 768 --layers 12 --seed 0 --sdm-big-mult 2"
arm sw_k64_d768_L12_50M        onesdm_allsdm     "--d 768 --layers 12 --seed 0 --sdm-k 64"
arm sw_heads8_d768_L12_50M     onesdm_allsdm     "--d 768 --layers 12 --seed 0 --sdm-heads 8"
arm sw_diary4k_d768_L12_50M    onesdm_allsdm     "--d 768 --layers 12 --seed 0 --mem-n-sub 64"
arm sw_deep_d512_L24_50M       onesdm_allsdm     "--d 512 --layers 24 --seed 0"
arm sw_wide_d1024_L8_50M       onesdm_allsdm     "--d 1024 --layers 8 --seed 0"
say "SWEEP DONE"
