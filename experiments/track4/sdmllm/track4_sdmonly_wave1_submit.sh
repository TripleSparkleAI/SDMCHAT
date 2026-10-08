#!/bin/bash
# SDMONLY wave 1: submit the sealed arms of PLAN_SDMONLY_TWO_DAY_PUSH_2026-10-03.md section 4 to the Spark queue.
# Waits for the speed bench to finish first, so the bench numbers are taken on a quiet GPU.
cd ~/settle24/runs/sdmonly
while ! grep -q BENCH_DONE bench.out; do sleep 10; done
Q="python3 $HOME/settle24/code/q.py --root $HOME/settle24/q"
A=$HOME/settle24/code/sdmonly_arm.sh
F="--eval-every 500 --ckpt-every 500 --log-every 100"
W="--n-back 8 --decays 0.5,0.8,0.9,0.97,0.99"
sub() { name=$1; shift; $Q submit -p 9 --threads 2 --gpu 1 --mem 14 --name "SDMONLY-$name" -- nice -n 10 $A "$@" $F; }
sub p0_A_c     new p0_A_c     sdmonly       0 20000000 train
sub p0_R_S     s0  p0_R_S     sdm           0 20000000 train $W --softness 0.25 --qgrad 0 --value-init zero --store-lr-mult 3 --store-wd 0
sub p0_R_N     s0  p0_R_N     sdm_nostore   0 20000000 train $W
sub p0_A_none  new p0_A_none  sdmonly_none  0 20000000 train
sub p0_A_dense new p0_A_dense sdmonly_dense 0 20000000 train
sub p0_R_Q     s0  p0_R_Q     qwen          0 20000000 train
sub p0_A_lr1    new p0_A_lr1    sdmonly 0 20000000 train --lr 1e-3
sub p0_A_lr6    new p0_A_lr6    sdmonly 0 20000000 train --lr 6e-3
sub p0_A_lr10   new p0_A_lr10   sdmonly 0 20000000 train --lr 1e-2
sub p0_A_slr1   new p0_A_slr1   sdmonly 0 20000000 train --store-lr-mult 1
sub p0_A_slr10  new p0_A_slr10  sdmonly 0 20000000 train --store-lr-mult 10
sub p0_A_h2     new p0_A_h2     sdmonly 0 20000000 train --hops 2
sub p0_A_h8     new p0_A_h8     sdmonly 0 20000000 train --hops 8
sub p0_A_n128   new p0_A_n128   sdmonly 0 20000000 train --n-sub 128
sub p0_A_n512   new p0_A_n512   sdmonly 0 20000000 train --n-sub 512
sub p0_A_k16    new p0_A_k16    sdmonly 0 20000000 train --k 16
sub p0_A_k64    new p0_A_k64    sdmonly 0 20000000 train --k 64
sub p0_A_heads4 new p0_A_heads4 sdmonly 0 20000000 train --heads 4
sub p0_A_share  new p0_A_share  sdmonly 0 20000000 train --share-store
sub p0_A_soft5  new p0_A_soft5  sdmonly 0 20000000 train --softness 0.5
sub p0_A_qg1    new p0_A_qg1    sdmonly 0 20000000 train --qgrad 1
sub p0_A_ro     new p0_A_ro     sdmonly 0 20000000 train --readout-f 688
sub p0_A_wsd    new p0_A_wsd    sdmonly 0 20000000 train --sched wsd
sub p0_A_wd     new p0_A_wd     sdmonly 0 20000000 train --store-wd 0.1
echo WAVE1_SUBMITTED $(date -u +%FT%TZ)
