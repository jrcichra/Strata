#!/bin/bash
# Pair --chunk (sizes the SWA KV ring margin) with STRATA_MAX_PREFILL_CHUNK (defeats the post-load cap),
# then verify long-prompt logits parity and prefill throughput.
export LD_LIBRARY_PATH=/mnt/win/llm-temp/enginetest
MODEL=/mnt/win/llm-temp/strata-models/models/inkling-2-mini/Inkling-2-Mini-SQ-10.strata
P=/mnt/win/llm-temp/scratch/parity
cd /mnt/win/llm-temp/build-ink/src/inkling2

python3 - <<'PY'
ids=[i.strip() for i in open('/mnt/win/llm-temp/scratch/parity/prompt256_cma.ids').read().split(',') if i.strip()]
open('/mnt/win/llm-temp/scratch/parity/prompt2048_cma.ids','w').write(','.join(ids*8))
print('prompt2048_cma.ids:', 8*len(ids), 'ids')
PY

echo "=== phase 1: reference logits at DEFAULTS (chunk 256, ring 1536) over 2048 positions ==="
unset STRATA_MAX_PREFILL_CHUNK
./strata-inkling --native "$MODEL" --parallel 1 --kv fp8 --vram-experts-gb 8.77 --ram-experts-gb 0 \
    --prompt-ids @"$P/prompt2048_cma.ids" --dump-ref "$P/ref_2048_def.bin" 2>&1 | grep -iE "ring|Capped" | head -2
ls -la "$P/ref_2048_def.bin" | awk '{print "ref bytes:", $5}'

echo "=== phase 2: --chunk 1024 + STRATA_MAX_PREFILL_CHUNK=1024 ==="
export STRATA_MAX_PREFILL_CHUNK=1024
./strata-inkling --native "$MODEL" --parallel 1 --kv fp8 --vram-experts-gb 8.77 --ram-experts-gb 0 \
    --chunk 1024 --selfcheck 2>&1 | grep -iE "ring|selfcheck|max_prefill" | head -3
echo "--- parity at 2048 positions (must be Mean KLD=0.000000) ---"
./strata-inkling --native "$MODEL" --parallel 1 --kv fp8 --vram-experts-gb 8.77 --ram-experts-gb 0 \
    --chunk 1024 --prompt-ids @"$P/prompt2048_cma.ids" --kld-eval "$P/ref_2048_def.bin" 2>&1 | grep kld-eval
echo "--- perf ---"
./strata-inkling --bench --native "$MODEL" --parallel 1 --kv fp8 --vram-experts-gb 8.77 --ram-experts-gb 0 \
    --chunk 1024 --prompt-len 2048 --gen 64 2>&1 | grep '\[bench\]'
./strata-inkling --bench --native "$MODEL" --parallel 1 --kv fp8 --vram-experts-gb 8.77 --ram-experts-gb 0 \
    --chunk 1024 --prompt-len 8192 --gen 32 2>&1 | grep '\[bench\]'

echo "=== phase 3: --chunk 2048 + STRATA_MAX_PREFILL_CHUNK=2048 ==="
export STRATA_MAX_PREFILL_CHUNK=2048
./strata-inkling --native "$MODEL" --parallel 1 --kv fp8 --vram-experts-gb 8.77 --ram-experts-gb 0 \
    --chunk 2048 --selfcheck 2>&1 | grep -iE "ring|selfcheck|max_prefill" | head -3
./strata-inkling --bench --native "$MODEL" --parallel 1 --kv fp8 --vram-experts-gb 8.77 --ram-experts-gb 0 \
    --chunk 2048 --prompt-len 8192 --gen 32 2>&1 | grep -E '\[bench\]|rror|memory' | head -2
echo "=== DONE ==="
