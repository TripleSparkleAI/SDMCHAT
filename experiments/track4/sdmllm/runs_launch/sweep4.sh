#!/bin/bash
# THE FULL SHAPE SWEEP, WAVE SWD (sealed 2026-10-09 in THE LOG before it fires): wave SWB left depth a tie inside the
# noise (d512 L32 1.48813, L48 1.48134), so rule (c) reruns the pair with more data (100M tokens), on the BEST diary
# of waves SWB and SWC (the lowest TEST among diary 256, 121 and 64 at d768 L12), with the 12-layer shape as the
# reference. Same recipe as waves SW, SWB and SWC, seed 0.
export SDMLLM_DATA=$HOME/sdmonly_base/data32k SDMLLM_CKPT=$HOME/sdmonly_base/launch/ck SDMLLM_RUNS=$HOME/sdmonly_base/launch/runs
export SDM_EVAL_TOKENS=4096 OMP_NUM_THREADS=8
P=$HOME/settle24/venv/bin/python; L=$HOME/sdmonly_base/launch/logs
cd $HOME/sdmonly_base/code
REC="--T 2048 --B 64 --accum 8 --lr 3e-3 --muon-lr 0.02 --sched wsd --compile-body --loss chunked --tokens 100000000 --train-name train_big_p200m --seed 0"
say() { echo "$(date -u +%FT%TZ) $*"; }
# the best diary: lowest TEST bpb among the three d768 L12 diary runs that have a result
NSUB=$($P - "$SDMLLM_RUNS" <<'PY'
import json, os, sys
runs = sys.argv[1]
best = None
for name, nsub in (("swb_diary256_d768_L12_50M", 16), ("swc_diary121_d768_L12_50M", 11), ("swc_diary64_d768_L12_50M", 8)):
    f = os.path.join(runs, name + ".result.json")
    if os.path.exists(f):
        b = json.load(open(f))["val_test"]["bpb"]
        if best is None or b < best[0]:
            best = (b, nsub)
print(best[1] if best else 16)
PY
)
SLOTS=$((NSUB * NSUB))
say "best diary: n_sub $NSUB ($SLOTS slots a head)"
arm() {  # run [flags]
  if [ -f $SDMLLM_RUNS/$1.result.json ]; then say "skip $1 (result exists)"; return; fi
  say "start $1"; $P track4_sdmonly_train.py --run $1 --arm onesdm_allsdm $REC $2 > $L/$1.out 2>&1; say "end $1 exit $?"
}
arm swd_shallow_diary${SLOTS}_d768_L12_100M "--d 768 --layers 12 --mem-n-sub $NSUB"
# addendum sealed 2026-10-09T15:03Z: diary 121 and 64 tied at 50M (1.40333 vs 1.40020), so rule (c) also reruns the
# other of the pair at 100M on the 12-layer shape
if [ "$NSUB" = 8 ]; then ONS=11; else ONS=8; fi
arm swd_shallow_diary$((ONS * ONS))_d768_L12_100M "--d 768 --layers 12 --mem-n-sub $ONS"
arm swd_deep32_diary${SLOTS}_d512_L32_100M  "--d 512 --layers 32 --mem-n-sub $NSUB"
arm swd_deep48_diary${SLOTS}_d512_L48_100M  "--d 512 --layers 48 --mem-n-sub $NSUB"
say "SWEEP4 DONE"
