#!/bin/bash
# A) measure 9 waves/SIMD variant; B) restore committed 8-wave build and smoke-test long context.
SRC=/mnt/win/llm-temp/strata-src
REL=engine/src/inkling2/hip/ink_hip_codecs.hip.h
H=$SRC/$REL
BIN=/mnt/win/llm-temp/build-ink/src/inkling2/strata-inkling
MODEL=/mnt/win/llm-temp/strata-models/models/inkling-2-mini/Inkling-2-Mini-SQ-10.strata
P=/mnt/win/llm-temp/scratch/parity
export LD_LIBRARY_PATH=/mnt/win/llm-temp/enginetest

restore() { cd "$SRC" && git checkout -- "$REL"; }
trap 'restore; echo "[tree restored to committed 8-wave state]"' EXIT
run() { cd /mnt/win/llm-temp/build-ink/src/inkling2 && "$@"; }

echo "=== A: __launch_bounds__(256, 9)  [112 VGPR, 20 spills, 84 B/lane] ==="
sed -i 's/__launch_bounds__(256, 8) k_moe_grouped_w13_situ(/__launch_bounds__(256, 9) k_moe_grouped_w13_situ(/' "$H"
grep -c "launch_bounds__(256, 9) k_moe_grouped_w13_situ" "$H"
cmake --build /mnt/win/llm-temp/build-ink --target strata-inkling -j 4 2>&1 | tail -2
run ./strata-inkling --native "$MODEL" --parallel 1 --kv fp8 --vram-experts-gb 8.77 --ram-experts-gb 0 \
    --prompt-ids @"$P/prompt256_cma.ids" --kld-eval "$P/ref_n256.bin" 2>&1 | grep kld-eval
for i in 1 2; do
  run ./strata-inkling --bench --native "$MODEL" --parallel 1 --kv fp8 --vram-experts-gb 8.77 \
      --ram-experts-gb 0 --prompt-len 2048 --gen 64 2>&1 | grep '\[bench\]'
done
STRATA_HIP_PROF=2 run ./strata-inkling --bench --native "$MODEL" --parallel 1 --kv fp8 \
    --vram-experts-gb 8.77 --ram-experts-gb 0 --prompt-len 2048 --gen 16 2>&1 | grep -oE "moe_w13=[0-9.]+" | head -4

echo "=== B: restore committed 8-wave build ==="
restore
cmake --build /mnt/win/llm-temp/build-ink --target strata-inkling -j 4 2>&1 | tail -2
md5sum "$H" "$BIN"
run ./strata-inkling --selfcheck --native "$MODEL" --parallel 1 --kv fp8 --vram-experts-gb 8.77 \
    --ram-experts-gb 0 2>&1 | grep '\[selfcheck\]'
echo "=== long-context smoke: prompt 8192 / gen 32 ==="
run ./strata-inkling --bench --native "$MODEL" --parallel 1 --kv fp8 --vram-experts-gb 8.77 \
    --ram-experts-gb 0 --prompt-len 8192 --gen 32 2>&1 | grep '\[bench\]'
echo "=== confirm committed state perf: prompt 2048 / gen 64 ==="
run ./strata-inkling --bench --native "$MODEL" --parallel 1 --kv fp8 --vram-experts-gb 8.77 \
    --ram-experts-gb 0 --prompt-len 2048 --gen 64 2>&1 | grep '\[bench\]'
echo "=== DONE ==="
