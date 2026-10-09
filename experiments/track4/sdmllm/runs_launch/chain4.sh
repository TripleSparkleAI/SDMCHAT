#!/bin/bash
# chain4.sh <sweep2 pid>: wait for wave SWB to finish, then fire wave SWC.
while kill -0 "$1" 2>/dev/null; do sleep 60; done
echo "$(date -u +%FT%TZ) sweep2 gone, start wave SWC"
bash $HOME/sdmonly_base/launch/sweep3.sh
