#!/bin/bash
# Validate the two-rows-per-warp grouped-w13 change: build, logits parity, bench x2, selfcheck.
set -o pipefail
SRC=/mnt/win/llm-temp/strata-src
BIN=/mnt/win/llm-temp/build-ink/src/inkling2/strata-inkling
MODEL=/mnt/win/llm-temp/strata-models/models/inkling-2-mini/Inkling-2-Mini-SQ-10.strata
P=/mnt/win/llm-temp/scratch/parity
export LD_LIBRARY_PATH=/mnt/win/llm-temp/enginetest

echo "=== md5 of codec header under test ==="
md5sum "$SRC/engine/src/inkling2/hip/ink_hip_codecs.hip.h"

echo "=== build ==="
cmake --build /mnt/win/llm-temp/build-ink --target strata-inkling -j 4 2>&1 | tail -25
BR=$?
echo "build rc=$BR"
[ $BR -ne 0 ] && { echo "BUILD FAILED - aborting"; exit 1; }
md5sum "$BIN"

run() { cd /mnt/win/llm-temp/build-ink/src/inkling2 && "$@"; }

echo "=== parity vs staging-off reference (must be KLD=0.000000 / 100.00% / PPL=929388.622) ==="
run ./strata-inkling --native "$MODEL" --parallel 1 --kv fp8 \
    --vram-experts-gb 8.77 --ram-experts-gb 0 \
    --prompt-ids @"$P/prompt256_cma.ids" --kld-eval "$P/ref_n256.bin" 2>&1 | tail -4
echo "parity rc=$?"

echo "=== bench x2 ==="
for i in 1 2; do
  run ./strata-inkling --bench --native "$MODEL" --parallel 1 --kv fp8 \
      --vram-experts-gb 8.77 --ram-experts-gb 0 --prompt-len 2048 --gen 64 2>&1 | grep '\[bench\]'
done

echo "=== selfcheck ==="
run ./strata-inkling --selfcheck --native "$MODEL" --parallel 1 --kv fp8 \
    --vram-experts-gb 8.77 --ram-experts-gb 0 2>&1 | grep -E '\[selfcheck\]'

echo "=== moe_w13 profile (shipped reference was ~871-895 ms/chunk) ==="
STRATA_HIP_PROF=2 run ./strata-inkling --bench --native "$MODEL" --parallel 1 --kv fp8 \
    --vram-experts-gb 8.77 --ram-experts-gb 0 --prompt-len 2048 --gen 16 2>&1 | grep -iE 'moe_w13|prefill' | head -8
echo "=== DONE ==="
