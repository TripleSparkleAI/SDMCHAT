#!/bin/bash
# track4_sdmonly_vast_watch_fleet11.sh - the fleet-11 pull loop and its deadline (a copy of
# track4_sdmonly_vast_watch_param.sh whose only change is the allowed deadline window, 0.2 to 7.7 hours, for the
# 7-hour fleet). Run detached on the M5 with WATCH_LIST and DEADLINE_UTC set:
#   WATCH_LIST=vast_sdmonly_fleet11.list DEADLINE_UTC=... nohup bash track4_sdmonly_vast_watch_fleet11.sh > runs_vast_sdmonly/watch_fleet11.out 2>&1 &
# Every 5 minutes it pulls each box's result json, logs and per-window files home (small files only; no weights:
# these are test runs). Each pull tries the direct IP, then the Vast ssh proxy. At the deadline it pulls once more and destroys BY ID only the boxes whose final pull succeeded; any other box is HELD and logged (pull before destroy).
# The deadline is an epoch computed in UTC by python (never macOS `date -j -f`, which reads local time and once
# destroyed a box nine hours early), and the script refuses to start unless the deadline is 0.2 to 7.7 hours away.
set -u
cd "$(dirname "$0")"
LIST=${WATCH_LIST:-vast_sdmonly_boxes.list}; OUT=runs_vast_sdmonly; V=$HOME/.vast-venv/bin/vastai
DEADLINE_UTC=${DEADLINE_UTC:-2026-10-03T15:45:00Z}
DL=$(python3 -c "import calendar,time,sys; print(calendar.timegm(time.strptime(sys.argv[1],'%Y-%m-%dT%H:%M:%SZ')))" "$DEADLINE_UTC")
NOW=$(date +%s); LEFT=$(( DL - NOW ))
if [ "$LEFT" -lt 720 ] || [ "$LEFT" -gt 27720 ]; then echo "REFUSE: deadline $DEADLINE_UTC is $LEFT s away (allowed 720 to 27720)"; exit 2; fi
mkdir -p "$OUT"; OKF=$OUT/.pulled_ok
echo "$(date -u +%FT%TZ) watch start, deadline $DEADLINE_UTC, $LEFT s away"
pull_one() {  # pull_one LABEL HOST PORT: 0 when all three copies succeed
  S="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o LogLevel=ERROR -p $3"
  rsync -az -e "$S" --exclude='*.pt' "root@$2:/root/settle/runs/" "$OUT/$1/" 2>/dev/null \
    && rsync -az -e "$S" "root@$2:/root/settle/logs/" "$OUT/$1/logs/" 2>/dev/null \
    && rsync -az -e "$S" --include='*/' --include='test_per_window.npz' --exclude='*' "root@$2:/root/settle/ck/" "$OUT/$1/ck/" 2>/dev/null; }
pull_all() {  # the direct IP first, then the Vast ssh proxy (some boxes answer on only one route); ids that pulled go to $OKF
  : > "$OKF"
  "$V" show instances --raw 2>/dev/null | python3 -c "import sys,json; [print(i['id'], i.get('ssh_host'), i.get('ssh_port')) for i in json.load(sys.stdin)]" > "$OUT/.proxy_routes" 2>/dev/null
  grep -v '^#' "$LIST" | while read -r id label card host port rest; do
    [ -n "$id" ] || continue
    mkdir -p "$OUT/$label/logs" "$OUT/$label/ck"
    ph=$(awk -v i="$id" '$1==i {print $2}' "$OUT/.proxy_routes"); pp=$(awk -v i="$id" '$1==i {print $3}' "$OUT/.proxy_routes")
    if pull_one "$label" "$host" "$port" || { [ -n "$ph" ] && pull_one "$label" "$ph" "$pp"; }; then
      echo "$id" >> "$OKF"; echo "$(date -u +%FT%TZ) pulled $label results=$(ls "$OUT/$label"/*.result.json 2>/dev/null | wc -l | tr -d ' ')"
    else echo "$(date -u +%FT%TZ) PULL FAILED $label (direct and proxy)"; fi
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
  if ! grep -qx "$id" "$OKF"; then echo "$(date -u +%FT%TZ) HELD $id $label: the final pull failed, so the box is NOT destroyed (pull before destroy); pull it by hand, then destroy it"; continue; fi
  "$V" destroy instance "$id" -y 2>&1 | tail -1
  echo "$(date -u +%FT%TZ) destroyed $id $label"
done
sleep 20
"$V" show instances --raw 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print('instances left on the account:', [(i['id'], i.get('label')) for i in d])"
echo "$(date -u +%FT%TZ) watch end"
