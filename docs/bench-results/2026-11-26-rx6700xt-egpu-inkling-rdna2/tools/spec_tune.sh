#!/bin/bash
# Device retune of the speculative-decode window model. spec_cost (default 4.2) is "the MoE share of a
# verify step"; on a bandwidth-bound RDNA2 card with 8.77 GiB of experts that calibration is suspect.
# Decode-heavy window so decode tps dominates: prompt 2048, gen 256.
export LD_LIBRARY_PATH=/mnt/win/llm-temp/enginetest
MODEL=/mnt/win/llm-temp/strata-models/models/inkling-2-mini/Inkling-2-Mini-SQ-10.strata
cd /mnt/win/llm-temp/build-ink/src/inkling2
unset STRATA_MAX_PREFILL_CHUNK

b() { # label, extra flags...
  local L="$1"; shift
  local OUT
  OUT=$(./strata-inkling --bench --native "$MODEL" --parallel 1 --kv fp8 --vram-experts-gb 8.77 --ram-experts-gb 0 \
        --prompt-len 2048 --gen 256 "$@" 2>&1 | grep -E '\[bench\]|rror' | head -1)
  printf '%-28s %s\n' "$L" "$(echo "$OUT" | sed 's/.*| //')"
}

echo "########## spec-decode window sweep (prompt 2048, gen 256) ##########"
b "default(beam4,cost4.2)"
b "draft-off"                 --draft off
b "cost=1.5"                  --draft-cost 1.5
b "cost=2.5"                  --draft-cost 2.5
b "cost=8.0"                  --draft-cost 8.0
b "beam=1"                    --draft-beam 1
b "beam=1 cost=1.5"           --draft-beam 1 --draft-cost 1.5
b "plain=1.0"                 --draft-plain 1.0
b "adapt"                     --draft-adapt
echo "########## acceptance stats ##########"
./strata-inkling --native "$MODEL" --parallel 1 --kv fp8 --vram-experts-gb 8.77 --ram-experts-gb 0 \
    --prompt-ids @/mnt/win/llm-temp/scratch/parity/prompt512_cma.ids --spec-check 2>&1 | grep -E "spec-check" | head -2
echo "  --draft off:"
./strata-inkling --native "$MODEL" --parallel 1 --kv fp8 --vram-experts-gb 8.77 --ram-experts-gb 0 \
    --prompt-ids @/mnt/win/llm-temp/scratch/parity/prompt512_cma.ids --spec-check --draft off 2>&1 | grep -E "spec-check" | head -2
echo "=== DONE ==="
