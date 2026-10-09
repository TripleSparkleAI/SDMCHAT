#!/bin/bash
# Waits for chain2 (the FULL chain, PID given) to finish, then runs wave SWB. Run in tmux on the Spark.
while kill -0 "$1" 2>/dev/null; do sleep 60; done
echo "$(date -u +%FT%TZ) chain2 gone, start wave SWB"
bash $HOME/sdmonly_base/launch/sweep2.sh
