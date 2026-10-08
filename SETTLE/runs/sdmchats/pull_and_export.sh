#!/bin/bash
# pull_and_export.sh RUN [KEY] - lane SDMCHATS: bring one finished Spark run home and (with KEY) export it for the site.
#   results: ~/settle24/runs/main/RUN.{result.json,log} -> experiments/track4/sdmllm/runs_sdmchats/ (committed)
#   checkpoint: ~/settle24/ck/main/RUN/last.pt -> <main checkout>/experiments/track4/sdmllm/checkpoints/RUN/ (gitignored)
#   export: tools/export_sdm_chat.py with SDMCHAT_ONLY=KEY into this worktree's public/data/sdmchat/ (gitignored)
# Example: bash pull_and_export.sh sdmchats_SW_d512_s0_300M sdmwide512
set -euo pipefail
RUN=$1; KEY=${2:-}
HERE="$(cd "$(dirname "$0")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"
MAIN="$(cd "$WT" && git rev-parse --path-format=absolute --git-common-dir | xargs dirname)"
mkdir -p "$WT/experiments/track4/sdmllm/runs_sdmchats" "$MAIN/experiments/track4/sdmllm/checkpoints/$RUN"
scp -q "spark:settle24/runs/main/$RUN.result.json" "spark:settle24/runs/main/$RUN.log" "$WT/experiments/track4/sdmllm/runs_sdmchats/"
python3 - "$WT/experiments/track4/sdmllm/runs_sdmchats/$RUN.result.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print("TEST", r["val_test"]["bpb"], "CHAT", r.get("val_extra", {}).get("chat_test", {}).get("bpb"),
      {k: v["bpb"] for k, v in r.items() if k.startswith("val_test_ablate")})
PY
[ -z "$KEY" ] && exit 0
scp -q "spark:settle24/ck/main/$RUN/last.pt" "$MAIN/experiments/track4/sdmllm/checkpoints/$RUN/last.pt"
cp "$WT/experiments/track4/sdmllm/runs_sdmchats/$RUN.result.json" "$MAIN/experiments/track4/sdmllm/checkpoints/$RUN/"
cd "$WT/SETTLE/settle-site"
SDMCHAT_ONLY=$KEY python3 tools/export_sdm_chat.py
