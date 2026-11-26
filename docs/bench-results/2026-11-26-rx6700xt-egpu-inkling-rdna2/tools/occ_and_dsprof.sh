#!/bin/bash
# Two things, one GPU session:
#  A/B: does forcing k_moe_direct_w13_situ from 12 to 16 waves/SIMD (at the cost of
#       8 spilled VGPRs) help decode? Decode-only kernel: B<=4 path, so prefill should
#       be flat and serves as a sanity control.
#  C:   parity gate. Occupancy attributes cannot reorder arithmetic, so KLD must be 0.
#  D:   is the DSpark draft controller stuck? Prints K (drafts/step) and per-draft device
#       cost at short vs long context. Caveat: DSPARK_PROF also disables the draft graph,
#       so absolute tot is inflated; the 512-vs-2048 comparison is still apples-to-apples.
set -u
BIN=/mnt/win/llm-temp/build-ink/src/inkling2/strata-inkling
MODEL=/mnt/win/llm-temp/strata-models/models/inkling-2-mini/Inkling-2-Mini-SQ-10.strata
P=/mnt/win/llm-temp/scratch/parity
export LD_LIBRARY_PATH=/mnt/win/llm-temp/enginetest
COMMON="--native $MODEL --parallel 1 --kv fp8 --vram-experts-gb 8.77 --ram-experts-gb 0"
run() { cd /mnt/win/llm-temp/build-ink/src/inkling2 && "$@"; }

echo "=== binary under test ==="; md5sum "$BIN"

echo "########## A) decode at prompt 2048, gen 256 steady state (bar: 51.0 tps) ##########"
for i in 1 2; do run ./strata-inkling --bench $COMMON --prompt-len 2048 --gen 256 2>&1 | grep '\[bench\]'; done

echo "########## B) prompt 8192 gen 512 (bar: 41.9 tps) ##########"
run ./strata-inkling --bench $COMMON --prompt-len 8192 --gen 512 2>&1 | grep '\[bench\]'

echo "########## C) parity gate: must be KLD=0.000000 ##########"
run ./strata-inkling $COMMON --prompt-ids @"$P/prompt512_cma.ids" --kld-eval "$P/ref_512_c256.bin" 2>&1 | grep kld-eval

echo "########## D) DSpark controller trajectory: K and draft cost ##########"
for PL in 512 2048; do
  STRATA_HIP_DSPARK_PROF=1 run ./strata-inkling --bench $COMMON --prompt-len $PL --gen 64 2>&1 \
    | grep 'DSPARK-PROF' > /tmp/dsp_$PL.txt
  echo "--- prompt $PL (lines=$(wc -l < /tmp/dsp_$PL.txt)) ---"
  echo "K histogram (drafts per step):"; grep -oE 'K=[0-9]+' /tmp/dsp_$PL.txt | sort | uniq -c
  awk '{for(i=1;i<=NF;i++) if($i ~ /^tot=/){gsub("tot=","",$i); s+=$i; n++; if($i+0>m) m=$i+0}}
       END{if(n) printf "draft device cost: mean=%.2fms max=%.2fms n=%d\n", s/n, m, n}' /tmp/dsp_$PL.txt
  run ./strata-inkling --bench $COMMON --prompt-len $PL --gen 256 2>&1 | grep '\[bench\]' | sed 's/^/  gen256 /'
done

echo "=== DONE ==="
