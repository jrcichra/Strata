#!/bin/bash
# Does a 512-token prefill chunk beat 256? Measures perf AND whether batch shape changes output.
SRC=/mnt/win/llm-temp/strata-src
REL=engine/src/inkling2/hip/ink_hip_backend.hip.cpp
BIN=/mnt/win/llm-temp/build-ink/src/inkling2/strata-inkling
MODEL=/mnt/win/llm-temp/strata-models/models/inkling-2-mini/Inkling-2-Mini-SQ-10.strata
P=/mnt/win/llm-temp/scratch/parity
export LD_LIBRARY_PATH=/mnt/win/llm-temp/enginetest

restore() { cd "$SRC" && git checkout -- "$REL"; }
trap 'restore; echo "[source restored]"' EXIT
run() { cd /mnt/win/llm-temp/build-ink/src/inkling2 && "$@"; }
BUILD() { cmake --build /mnt/win/llm-temp/build-ink --target strata-inkling -j 4 2>&1 | tail -2; }

echo "=== phase 1: baseline build (chunk 256) -> reference logits for the 512-id prompt ==="
BUILD
run ./strata-inkling --native "$MODEL" --parallel 1 --kv fp8 --vram-experts-gb 8.77 --ram-experts-gb 0 \
    --prompt-ids @"$P/prompt512_cma.ids" --dump-ref "$P/ref_512_c256.bin" 2>&1 | grep -iE "capped|dump|ref" | tail -3
ls -la "$P/ref_512_c256.bin" 2>/dev/null | awk '{print "ref size:", $5}'
run ./strata-inkling --bench --native "$MODEL" --parallel 1 --kv fp8 --vram-experts-gb 8.77 \
    --ram-experts-gb 0 --prompt-len 2048 --gen 64 2>&1 | grep '\[bench\]'

echo "=== phase 2: allow chunk 512 at our free VRAM (982 MiB) ==="
sed -i 's/const int tight_chunk = (post_free < (1536ULL << 20)) ? 256 : 512;/const int tight_chunk = (post_free < (900ULL << 20)) ? 256 : 512;/' "$SRC/$REL"
grep -c "900ULL << 20" "$SRC/$REL"
BUILD
echo "--- confirm the chunk actually became 512 ---"
run ./strata-inkling --selfcheck --native "$MODEL" --parallel 1 --kv fp8 --vram-experts-gb 8.77 \
    --ram-experts-gb 0 2>&1 | grep -iE "capped|selfcheck"

echo "--- numerics vs the chunk-256 reference (512-id prompt; KLD=0 means shape-invariant) ---"
run ./strata-inkling --native "$MODEL" --parallel 1 --kv fp8 --vram-experts-gb 8.77 --ram-experts-gb 0 \
    --prompt-ids @"$P/prompt512_cma.ids" --kld-eval "$P/ref_512_c256.bin" 2>&1 | grep kld-eval

echo "--- perf at chunk 512 ---"
for i in 1 2; do
  run ./strata-inkling --bench --native "$MODEL" --parallel 1 --kv fp8 --vram-experts-gb 8.77 \
      --ram-experts-gb 0 --prompt-len 2048 --gen 64 2>&1 | grep '\[bench\]'
done
run ./strata-inkling --bench --native "$MODEL" --parallel 1 --kv fp8 --vram-experts-gb 8.77 \
    --ram-experts-gb 0 --prompt-len 8192 --gen 32 2>&1 | grep '\[bench\]'
echo "=== DONE ==="
