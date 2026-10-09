#!/bin/bash
# chain6.sh <chain5 pid>: wait for wave SWD to finish, then run the 100M yardstick.
while kill -0 "$1" 2>/dev/null; do sleep 60; done
echo "$(date -u +%FT%TZ) chain5 gone, start the 100M yardstick"
bash $HOME/sdmonly_base/launch/sweep5.sh
