#!/bin/bash
# track4_sdmonly_vast_watch.sh - the test fleet's pull loop and its deadline. Run detached on the M5:
#   nohup bash track4_sdmonly_vast_watch.sh > runs_vast_sdmonly/watch.out 2>&1 &
# Every 5 minutes it pulls each box's result json, logs and per-window files home (small files only; no weights:
# these are test runs). At the deadline it pulls once more and then destroys every box in the list BY ITS ID.
# The deadline is an epoch computed in UTC by python (never macOS `date -j -f`, which reads local time and once
# destroyed a box nine hours early), and the script refuses to start unless the deadline is 0.2 to 5.2 hours away.
set -u
cd "$(dirname "$0")"
LIST=vast_sdmonly_boxes.list; OUT=runs_vast_sdmonly; V=$HOME/.vast-venv/bin/vastai
DEADLINE_UTC=${DEADLINE_UTC:-2026-10-03T15:45:00Z}
DL=$(python3 -c "import calendar,time,sys; print(calendar.timegm(time.strptime(sys.argv[1],'%Y-%m-%dT%H:%M:%SZ')))" "$DEADLINE_UTC")
NOW=$(date +%s); LEFT=$(( DL - NOW ))
if [ "$LEFT" -lt 720 ] || [ "$LEFT" -gt 18720 ]; then echo "REFUSE: deadline $DEADLINE_UTC is $LEFT s away (allowed 720 to 18720)"; exit 2; fi
mkdir -p "$OUT"
echo "$(date -u +%FT%TZ) watch start, deadline $DEADLINE_UTC, $LEFT s away"
pull_all() {
  grep -v '^#' "$LIST" | while read -r id label card host port rest; do
    [ -n "$id" ] || continue
    S="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o LogLevel=ERROR -p $port"
    mkdir -p "$OUT/$label/logs" "$OUT/$label/ck"
    rsync -az -e "$S" --exclude='*.pt' "root@$host:/root/settle/runs/" "$OUT/$label/" 2>/dev/null \
      && rsync -az -e "$S" "root@$host:/root/settle/logs/" "$OUT/$label/logs/" 2>/dev/null \
      && rsync -az -e "$S" --include='*/' --include='test_per_window.npz' --exclude='*' "root@$host:/root/settle/ck/" "$OUT/$label/ck/" 2>/dev/null \
      && echo "$(date -u +%FT%TZ) pulled $label results=$(ls "$OUT/$label"/*.result.json 2>/dev/null | wc -l | tr -d ' ')" \
      || echo "$(date -u +%FT%TZ) PULL FAILED $label"
  done
}
while [ "$(date +%s)" -lt "$DL" ]; do
  pull_all
  REM=$(( DL - $(date +%s) )); [ "$REM" -gt 300 ] && sleep 300 || { [ "$REM" -gt 0 ] && sleep "$REM"; }
done
echo "$(date -u +%FT%TZ) DEADLINE: final pull, then destroy"
pull_all
grep -v '^#' "$LIST" | while read -r id label rest; do
  [ -n "$id" ] || continue
  "$V" destroy instance "$id" -y 2>&1 | tail -1
  echo "$(date -u +%FT%TZ) destroyed $id $label"
done
sleep 20
"$V" show instances --raw 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print('instances left on the account:', [(i['id'], i.get('label')) for i in d])"
echo "$(date -u +%FT%TZ) watch end"
