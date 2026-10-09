#!/bin/bash
# unfreeze.sh PID...: every 60 s, if a listed PID is stopped (state T), send it SIGCONT and log it. Explicit PIDs only.
# Born 2026-10-09: sweep2.sh was found stopped (T) twice with no known sender, each time stalling the sweep.
L=$HOME/sdmonly_base/launch/unfreeze.log
while :; do
  alive=0
  for p in "$@"; do
    s=$(awk '/^State:/{print $2}' /proc/$p/status 2>/dev/null) || continue
    [ -n "$s" ] && alive=1
    [ "$s" = "T" ] && { kill -CONT "$p"; echo "$(date -u +%FT%TZ) CONT $p" >> $L; }
  done
  [ $alive = 1 ] || exit 0
  sleep 60
done
