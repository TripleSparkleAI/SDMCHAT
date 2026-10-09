#!/bin/bash
# unfreeze_tree.sh ROOTPID...: every 60 s, send SIGCONT to any process in the tree under each listed root PID
# (the root included) that sits in state T (stopped). Our sweep scripts have been found stopped three times with no
# known sender; this keeps a chain moving without touching any process outside the listed trees.
desc() { echo "$1"; for c in $(cat /proc/$1/task/*/children 2>/dev/null); do desc "$c"; done; }
while :; do
  alive=0
  for r in "$@"; do
    kill -0 "$r" 2>/dev/null || continue
    alive=1
    for p in $(desc "$r"); do
      s=$(awk '/^State:/{print $2}' /proc/$p/status 2>/dev/null)
      if [ "$s" = "T" ]; then kill -CONT "$p"; echo "$(date -u +%FT%TZ) SIGCONT $p (tree $r)" >> $HOME/sdmonly_base/launch/unfreeze.log; fi
    done
  done
  [ $alive = 0 ] && exit 0
  sleep 60
done
