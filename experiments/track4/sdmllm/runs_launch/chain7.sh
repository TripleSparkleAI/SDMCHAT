#!/bin/bash
# chain7.sh <chain6 pid>: wait for the 100M yardstick, then fire wave SWE.
while kill -0 "$1" 2>/dev/null; do sleep 60; done
echo "$(date -u +%FT%TZ) chain6 gone, start wave SWE"
bash $HOME/sdmonly_base/launch/sweep6.sh
