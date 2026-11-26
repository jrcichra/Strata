#!/bin/bash
# Separates "RDNA2 has no WMMA" from "the *_wmma kernels are better tiled".
# Off by default; STRATA_INK_MOE_TILED=1 puts grouped MoE on the tiled BMxBK LDS path on RDNA2.
# This path accumulates in a different order, so it is NOT expected to be bit-identical: measure KLD/PPL
# and hold for owner acceptance even if it is fast.
set -e
export LD_LIBRARY_PATH=/mnt/win/llm-temp/enginetest
MODEL=/mnt/win/llm-temp/strata-models/models/inkling-2-mini/Inkling-2-Mini-SQ-10.strata
P=/mnt/win/llm-temp/scratch/parity
cd /mnt/win/llm-temp/build-ink && cmake --build . --target strata-inkling -j 4 2>&1 | tail -3
cd src/inkling2
V() { ./strata-inkling --native "$MODEL" --parallel 1 --kv fp8 --vram-experts-gb 8.77 --ram-experts-gb 0 "$@" 2>&1; }

echo "=== A) gate OFF: must reproduce HEAD a0e388d (186.7-187.0 ppts, KLD=0, PPL=1101.388) ==="
unset STRATA_INK_MOE_TILED
V --selfcheck | grep -iE "selfcheck|Capped"
V --prompt-ids @"$P/prompt512_cma.ids" --kld-eval "$P/ref_512_c256.bin" | grep kld-eval
for i in 1 2; do V --bench --prompt-len 2048 --gen 64 | grep '\[bench\]'; done
echo "=== B) gate ON: perf first, then numerics only if perf moved ==="
export STRATA_INK_MOE_TILED=1
V --selfcheck | grep -iE "selfcheck|rror"
for i in 1 2; do V --bench --prompt-len 2048 --gen 64 | grep -E '\[bench\]|rror'; done
V --bench --prompt-len 8192 --gen 32 | grep -E '\[bench\]|rror'
V --prompt-ids @"$P/prompt512_cma.ids" --kld-eval "$P/ref_512_c256.bin" | grep -E "kld-eval|rror"
echo "=== DONE ==="
