#!/bin/bash
# track4_sdmonly_vast_pull_true.sh - the pull loop for the true training run (2026-10-04). It copies each box's small
# results home every 15 minutes (runs without *.pt, logs, ck/*/test_per_window.npz), trying the direct IP and then the
# Vast ssh proxy, exactly like track4_sdmonly_vast_watch_fleet11.sh. It NEVER destroys a box: the true runs last a day,
# so no deadline fits them; the hourly idle sweep closes a box only after it has been idle for an hour and pulled.
# It stops by itself when its list holds no box. Run detached on the M5:
#   nohup bash track4_sdmonly_vast_pull_true.sh >> runs_vast_sdmonly/pull_true.out 2>&1 &
set -u
cd "$(dirname "$0")"
LIST=${WATCH_LIST:-vast_sdmonly_true.list}; OUT=runs_vast_sdmonly; V=$HOME/.vast-venv/bin/vastai
mkdir -p "$OUT"
pull_one() {
  S="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o LogLevel=ERROR -p $3"
  rsync -az -e "$S" --exclude='*.pt' "root@$2:/root/settle/runs/" "$OUT/$1/" 2>/dev/null \
    && rsync -az -e "$S" "root@$2:/root/settle/logs/" "$OUT/$1/logs/" 2>/dev/null \
    && rsync -az -e "$S" --include='*/' --include='test_per_window.npz' --exclude='*' "root@$2:/root/settle/ck/" "$OUT/$1/ck/" 2>/dev/null; }
echo "$(date -u +%FT%TZ) pull loop start, list $LIST"
while grep -q '^[0-9]' "$LIST"; do
  "$V" show instances --raw 2>/dev/null | python3 -c "import sys,json; [print(i['id'], i.get('ssh_host'), i.get('ssh_port')) for i in json.load(sys.stdin)]" > "$OUT/.proxy_routes_true" 2>/dev/null
  grep -v '^#' "$LIST" | while read -r id label card host port rest; do
    [ -n "$id" ] || continue
    mkdir -p "$OUT/$label/logs" "$OUT/$label/ck"
    ph=$(awk -v i="$id" '$1==i {print $2}' "$OUT/.proxy_routes_true"); pp=$(awk -v i="$id" '$1==i {print $3}' "$OUT/.proxy_routes_true")
    if pull_one "$label" "$host" "$port" || { [ -n "$ph" ] && pull_one "$label" "$ph" "$pp"; }; then
      echo "$(date -u +%FT%TZ) pulled $label results=$(ls "$OUT/$label"/*.result.json 2>/dev/null | wc -l | tr -d ' ')"
    else echo "$(date -u +%FT%TZ) PULL FAILED $label (direct and proxy)"; fi
  done
  sleep 900
done
echo "$(date -u +%FT%TZ) list empty, pull loop end"
