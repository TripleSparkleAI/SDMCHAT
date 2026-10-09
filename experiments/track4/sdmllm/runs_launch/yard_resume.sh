#!/bin/bash
# HELD 2026-10-09T05:10Z by the navigator ("hold this transformer comparison till later"; SDM work first).
# Resumes the 1.1B transformer yardstick from its step-7,000 checkpoint (last.pt; copy hold_step7000.pt,
# sha256 81e790e28a20...). Same command as the last step of chain2, so the trainer resumes rather than restarts.
export SDMLLM_DATA=$HOME/sdmonly_base/data32k SDMLLM_CKPT=$HOME/sdmonly_base/launch/ck SDMLLM_RUNS=$HOME/sdmonly_base/launch/runs
export SDM_EVAL_TOKENS=4096 OMP_NUM_THREADS=8
P=$HOME/settle24/venv/bin/python; L=$HOME/sdmonly_base/launch/logs
cd $HOME/sdmonly_base/code
SHAPE="--d 768 --layers 12 --T 2048 --B 64 --accum 8 --lr 3e-3 --muon-lr 0.02 --sched wsd --compile-body --loss chunked --seed 0"
R=launch_yard_d768_L12_T2048_1100M
[ -f $SDMLLM_RUNS/$R.result.json ] && { echo "already finished"; exit 0; }
echo "$(date -u +%FT%TZ) resume $R"
$P track4_sdmonly_train.py --arm yardstick_transformer --run $R --tokens 1100000000 --train-name train_big $SHAPE --keep-at 550000000 >> $L/$R.out 2>&1
echo "$(date -u +%FT%TZ) end $R exit $?"
