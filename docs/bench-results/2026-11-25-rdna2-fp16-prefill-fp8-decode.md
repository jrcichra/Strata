# RX 6700 XT: FP16 prefill with FP8 decode KV

Date: 2026-11-25  
GPU: AMD Radeon RX 6700 XT (RDNA2, gfx1031)  
Model: Inkling-2-Mini-SQ-10  
Build: local HIP build from `rdna2-moe-lds-staging`  
Workload: 2,048 prompt tokens, 256 generated tokens, parallelism 1, 8.77 GiB VRAM experts, no RAM experts.

## Results

Three sequential runs per configuration, same binary and hardware. Hybrid conversion time is included in decode time.

| KV configuration | Prefill (tok/s), runs | Median prefill | Decode (tok/s), runs | Median decode |
|---|---:|---:|---:|---:|
| FP8 throughout | 186.4, 186.8, 186.9 | 186.8 | 53.7, 53.6, 53.6 | 53.6 |
| FP16 prefill → FP8 before decode | 208.2, 208.3, 207.9 | 208.2 | 57.0, 56.9, 56.9 | 56.9 |

Hybrid improved median prefill by 11.5% and decode by 6.2% for this workload.

## Numerical check

Teacher-forced evaluation after conversion against the existing FP8 reference: 255 positions, mean KLD 0.001667, top-1 agreement 100.00%. This validates next-token logits on the measured suffix; it is not a full-generation quality evaluation.

## Reproduction

Build the `strata-inkling` target, then run from `build-ink/src/inkling2` with the runtime libraries available:

```sh
export LD_LIBRARY_PATH=/mnt/win/llm-temp/enginetest
export STRATA_BENCH_FP8_DECODE=1
./strata-inkling --bench \
  --native /mnt/win/llm-temp/strata-models/models/inkling-2-mini/Inkling-2-Mini-SQ-10.strata \
  --parallel 1 --kv fp16 --vram-experts-gb 8.77 --ram-experts-gb 0 \
  --prompt-len 2048 --gen 256
```

Unset `STRATA_BENCH_FP8_DECODE` and use `--kv fp8` for the control. For the conversion parity check, use `--kld-eval /mnt/win/llm-temp/scratch/parity/ref_512_c256.bin` with the same model, parallelism, and expert placement, `--kv fp16`, and the environment variable set.

## Scope and limits

The FP16-to-FP8 switch is an experimental benchmark/research hook, not automatically enabled for serving. It currently rejects UMA, multi-GPU EP2, and host-offloaded KV pools. Conversion allocates FP8 buffers while the FP16 cache is still resident, so available VRAM limits the context length; the 8K FP16 case spilled KV to host and was not a successful hybrid run. Validate memory headroom and full-generation quality before considering serving integration.
