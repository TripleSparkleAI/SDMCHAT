#!/bin/bash
# Pull the launch chain's small records home from the Spark every 15 minutes: result and before json, run logs, the chain log.
cd "$(dirname "$0")"
while true; do
  rsync -a --timeout=60 -e "ssh -o BatchMode=yes -o ConnectTimeout=20" --include='*.json' --include='*.log' --exclude='*' spark:sdmonly_base/launch/runs/ ./ \
    && rsync -a --timeout=60 -e "ssh -o BatchMode=yes -o ConnectTimeout=20" spark:sdmonly_base/launch/chain.log ./chain.log \
    && echo "$(date -u +%FT%TZ) pulled $(ls *.result.json 2>/dev/null | wc -l | tr -d ' ') results" || echo "$(date -u +%FT%TZ) PULL FAILED"
  sleep 900
done
