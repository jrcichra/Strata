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

## Follow-up tuning on the resident build

`--bench-repeat N` now reuses the initialized backend and resident model weights, resets KV slot state between runs, and prints each result separately. This removes repeated model loading from an A/B cycle. It is incompatible with the experimental FP16-to-FP8 conversion switch because that switch changes the backend cache dtype. The first iteration may include one-time page growth/graph capture; use the later iterations for steady-state numbers.

Three same-process runs with FP8 KV, 8.77 GiB VRAM experts, 512-token chunk:

| Prompt / generation | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| 2,048 / 256 | 186.9, 187.4, 186.8 | 53.6, 53.9, 53.9 |
| 8,192 / 512 | 158.7, 158.8, 158.9 | 48.8, 49.7, 49.7 |

The first 8K decode includes cache growth/graph recapture; subsequent runs show 49.7 tok/s steady state.

### Rejected tuning candidates

| Candidate | Repeated result | Decision |
|---|---|---|
| W2 launch-bounds hint 12 waves | decode 52.4, 52.8, 52.8 tok/s | Regresses vs 16-wave hint |
| W2 hint 14 waves | decode 53.6, 53.9, 53.9 tok/s | Ties 16; retain known 16-wave setting |
| W2 hint 20 waves | decode 52.6, 52.9, 52.9 tok/s; compiler could only reach 12 waves | Regresses; target is infeasible |
| Grouped W13 hint 6 waves | prefill 173.3, 173.7, 173.8 tok/s | Regresses about 7% vs eight waves |
| Grouped W13 hint 7 waves | prefill 173.3, 173.8, 173.8 tok/s | Same regression; retain eight waves |
| Chunk 640 at 8K | two runs: prefill 160.0, 159.7; decode 22.4, 23.7 tok/s | Less than 1% prefill gain with about 52% decode loss; retain 512 |
| `--vram-experts-gb 8.5` | prefill 186.8, 187.1, 187.1; decode 53.6, 54.0, 54.0 tok/s | No measurable change vs 8.77 on this run |

The normal 2K eight-wave grouped-W13 control was 186.8–187.4 prefill tok/s. Normal W2 16-wave control was 53.6–53.9 decode tok/s. All runs use the same build and GPU unless noted above.

## Upstream version check

The checked-out source is `v0.2.21.1` plus the RDNA2 tuning branch; `strata-inkling` reports engine version 0.1.67. Upstream `v0.2.22` is newer, but its release notes explicitly keep AMD/Intel/Apple Inkling at 0.1.67 because those engines did not change. The upstream changes to Inkling engine paths since our base are the engine version bump/build-symbol work and a CUDA peer-card fix; no newer HIP kernel change was found. The 0.2.21 release lists an 8.5 GiB VRAM expert budget workaround for RX 6700 XT-class cards; testing that budget here did not improve the measured rates. See [0.2.21](https://github.com/Niko1221/Strata/releases/tag/v0.2.21) and [0.2.22](https://github.com/Niko1221/Strata/releases/tag/v0.2.22).
