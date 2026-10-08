#!/bin/bash
# track4_sdmonly_vast_setup.sh - run ON a rented box, from /root/settle/sdmllm. Idempotent: every stage leaves a
# marker in /root/settle/logs/ and is skipped when its marker exists.
# Stages: STAMP (driver, torch, cgroup cores, memory) -> DEPS (pip, gcc for torch.compile) -> DATA (rebuild S0's
# FineWeb-Edu shards on the box from Hugging Face and check their sha256 against the provenance file; the home
# link is about 0.5 Mbps, so shards are never copied) -> SELFTEST (the model's 15 checks on this GPU) -> READY.
# The canonical token_bytes.i32 and the tokenizer json come with the code (small files); the box-built
# token_bytes.i32 once differed at two ids (2026-10-01), so the canonical file is restored after the build.
set -u
R=/root/settle; L=$R/logs; D=$R/sdmllm/data
mkdir -p "$L" "$D" "$R/ck" "$R/runs"
cd "$R/sdmllm" || exit 1
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$L/setup.log"; }

if [ ! -f "$L/STAMPED" ]; then
  { echo "driver $(nvidia-smi --query-gpu=driver_version,name,memory.total --format=csv,noheader)"
    echo "torch $(python -c 'import torch; print(torch.__version__, torch.version.cuda)')"
    echo "nproc_all $(nproc --all)"
    echo "cpu_max $(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo unknown)"
    echo "memory_max $(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo unknown)"
    echo "disk $(df -h /root | tail -1)"; } > "$L/STAMPED.tmp" && mv "$L/STAMPED.tmp" "$L/STAMPED"
  say "STAMPED: $(tr '\n' ';' < "$L/STAMPED")"
fi

if [ ! -f "$L/DEPS" ]; then
  say "DEPS start"
  (apt-get update -qq && apt-get install -y -qq gcc g++ rsync > /dev/null) 2>> "$L/setup.log"
  pip install -q --no-input numpy pyarrow huggingface_hub tokenizers transformers 2>> "$L/setup.log" \
    && touch "$L/DEPS" && say "DEPS done" || { say "DEPS FAILED"; exit 1; }
fi

if [ ! -f "$L/DATA" ]; then
  say "DATA start (rebuild S0 shards from Hugging Face)"
  cp -n "$R/sdmllm/seed_data/ds4_v4flash_tokenizer.json" "$D/" 2>/dev/null || true
  python track4_sdmllm_prepare_fineweb_edu_token_shards.py > "$L/data_pipeline.log" 2>&1 || { say "DATA build FAILED"; exit 1; }
  cp "$R/sdmllm/seed_data/token_bytes.i32" "$D/token_bytes.i32"
  T=$(sha256sum "$D/train.u32" | cut -d' ' -f1); V=$(sha256sum "$D/val.u32" | cut -d' ' -f1); B=$(sha256sum "$D/token_bytes.i32" | cut -c1-8)
  say "sha train $T val $V token_bytes $B"
  if [ "$T" = "c2f7bc059eb399cad90a1ab38b2ecba0a679adc179d284aa0d34697dbdb9e5c7" ] && \
     [ "$V" = "89fbba2c788c0f15c4835cdc9197c035c1a9ad45365e83d3692c737653aa974f" ] && [ "$B" = "69f96025" ]; then
    touch "$L/DATA"; say "DATA ALL_MATCH"
  else say "DATA SHA MISMATCH"; exit 1; fi
fi

if [ ! -f "$L/SELFTEST" ]; then
  python track4_sdmonly_models.py --selftest > "$L/selftest.out" 2>&1 && touch "$L/SELFTEST"
  say "SELFTEST $(tail -1 "$L/selftest.out")"
  [ -f "$L/SELFTEST" ] || exit 1
fi
touch "$L/READY"; say "READY"
