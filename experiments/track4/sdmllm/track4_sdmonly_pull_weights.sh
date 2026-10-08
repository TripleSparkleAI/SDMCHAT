#!/bin/bash
# track4_sdmonly_pull_weights.sh LABEL HOST PORT RUN - wait for RUN's result json on the box, then copy its final
# checkpoint ck/RUN/last.pt home to runs_vast_sdmonly/LABEL/ck/RUN/ and check the size matches the box (the true runs'
# weights must come home before a box is destroyed; the pull loop skips *.pt). Polls every 5 minutes; never destroys.
set -u
cd "$(dirname "$0")"
L=$1; H=$2; P=$3; R=$4; OUT=runs_vast_sdmonly/$L/ck/$R
S="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o LogLevel=ERROR -p $P"
mkdir -p "$OUT"
until $S root@$H "test -f /root/settle/runs/$R.result.json" 2>/dev/null; do sleep 300; done
echo "$(date -u +%FT%TZ) $R finished; copying last.pt"
for i in 1 2 3; do
  rsync -a --partial -e "$S" "root@$H:/root/settle/ck/$R/last.pt" "$OUT/last.pt" && break; sleep 60; done
rb=$($S root@$H "stat -c %s /root/settle/ck/$R/last.pt" 2>/dev/null); lb=$(stat -f %z "$OUT/last.pt" 2>/dev/null)
[ -n "$rb" ] && [ "$rb" = "$lb" ] && echo "$(date -u +%FT%TZ) WEIGHTS HOME $R $lb bytes" || echo "$(date -u +%FT%TZ) WEIGHTS MISMATCH $R box=$rb home=$lb"
