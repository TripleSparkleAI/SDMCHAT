#!/bin/bash
# chain5.sh <chain4 pid>: wait for wave SWC to finish, then fire wave SWD.
while kill -0 "$1" 2>/dev/null; do sleep 60; done
echo "$(date -u +%FT%TZ) chain4 gone, start wave SWD"
bash $HOME/sdmonly_base/launch/sweep4.sh
