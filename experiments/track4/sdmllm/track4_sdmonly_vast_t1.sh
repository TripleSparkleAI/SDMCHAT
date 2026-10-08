#!/bin/bash
# track4_sdmonly_vast_t1.sh GRID - run ON a rented box after setup says READY. One GPU, one run at a time.
# 1. speed bench: the six configs the Spark bench ran (1.6M tokens each), so tokens a second compare card to card
# 2. anchor: wave 1's centre arm p0_A_c (20M tokens, seed 0), to compare with the Spark's own p0_A_c
# 3. the sizing grid cells named in GRID, each "d:tokens" (centre settings, wsd off), e.g. "512:20000000 512:60000000"
# A finished run (its result json exists) is skipped, so the script can be started again after a cut.
set -u
R=/root/settle; A=$R/sdmllm/track4_sdmonly_vast_arm.sh; L=$R/logs
while [ ! -f "$L/READY" ]; do sleep 10; done
run() { name=$1; shift; [ -f "$R/runs/$name.result.json" ] && { echo "skip $name"; return; }
  echo "$(date -u +%FT%TZ) start $name" >> "$L/t1.log"; bash "$A" "$@" > "$L/$name.out" 2>&1
  echo "$(date -u +%FT%TZ) end $name rc $?" >> "$L/t1.log"; }
F="--val-windows 8 --quick-val 4 --eval-every 100000 --ckpt-every 100000 --log-every 20 --fresh"
run bench_d256_n256_h4        new bench_d256_n256_h4        sdmonly       0 1600000 train --d 256 --n-sub 256 --hops 4 $F
run bench_d768_n256_h4        new bench_d768_n256_h4        sdmonly       0 1600000 train --d 768 --n-sub 256 --hops 4 $F
run bench_d768_n512_h4        new bench_d768_n512_h4        sdmonly       0 1600000 train --d 768 --n-sub 512 --hops 4 $F
run bench_d768_n512_h4_shared new bench_d768_n512_h4_shared sdmonly       0 1600000 train --d 768 --n-sub 512 --hops 4 --share-store $F
run bench_d768_n512_h8_shared new bench_d768_n512_h8_shared sdmonly       0 1600000 train --d 768 --n-sub 512 --hops 8 --share-store $F
run bench_d768_dense_h4       new bench_d768_dense_h4       sdmonly_dense 0 1600000 train --d 768 --n-sub 512 --hops 4 $F
echo "$(date -u +%FT%TZ) BENCH_DONE" >> "$L/t1.log"
G="--eval-every 500 --ckpt-every 500 --log-every 100"
run p0_A_c new p0_A_c sdmonly 0 20000000 train $G
echo "$(date -u +%FT%TZ) ANCHOR_DONE" >> "$L/t1.log"
for cell in $1; do d=${cell%%:*}; tok=${cell##*:}; name="sz_d${d}_$((tok/1000000))M"
  run "$name" new "$name" sdmonly 0 "$tok" train --d "$d" $G
done
echo "$(date -u +%FT%TZ) T1_DONE" >> "$L/t1.log"
