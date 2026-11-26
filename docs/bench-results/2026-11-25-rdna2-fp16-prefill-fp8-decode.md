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

### Grouped W13 expert-reuse tile

The grouped W13/SITU path originally handled up to 8 routed tokens per expert tile. A tile-16 variant retains the existing arithmetic order and stages X for 16 tokens while reusing each loaded expert-weight tile across them. Exact teacher-forced parity on 511 suffix positions was unchanged (mean KLD 0.000000, top-1 100%).

| Tile | 2K prompt, 256 generated (three runs) | 8K prompt, one run | Compiler occupancy | Decision |
|---|---:|---:|---:|---|
| 8 (prior control) | 186.8–187.4 prefill tok/s | 158.7–158.9 prefill tok/s (512 generated) | 8-wave target | Baseline |
| 12 | 187.5–187.9 prefill; 53.6–54.0 decode tok/s | — | No occupancy warning | Slightly below 16 |
| 16 | 189.3–189.9 prefill; 53.5–54.0 decode tok/s | 160.7 prefill tok/s (1 generated) | Met 8-wave target | Keep; about 1–2% prefill gain, decode unchanged |
| 24 | 185.5–186.1 prefill; 53.6–54.0 decode tok/s | — | No occupancy warning | Regresses vs 16 |
| 32 | 156.9–157.5 prefill; 53.7–53.9 decode tok/s | — | Fell to 4 waves from target 8 | Reject |

A one-shot `STRATA_HIP_PROF=2` sample at B=512 measured W13 stage time at 1,369 ms with tile-16 versus the earlier tile-8 sample at 1,411 ms; total profiled time moved from 2,336 ms to 2,293 ms. The profiler disables graph capture, so these numbers locate bottlenecks and are not throughput benchmarks. In that sample W13 remained the largest stage (~60%); qkvrg was next (~448 ms, ~20%).

### Driver and clock policy scan

The RX 6700 XT is `gfx1031` on kernel 7.3.0-9-generic / HIP 7.15. PCIe was at x16 Gen4, so bus bandwidth is not limiting the all-VRAM case. The driver was left at `power_dpm_force_performance_level=auto` with BOOTUP_DEFAULT profile. A temporary peak profile was slower (about 182 prefill / 53 decode tok/s vs about 187 / 54) and used more power; manual + COMPUTE profile tied auto within noise. Both experiments were restored. During a long run the GPU was busy ~99%, core clock ~2.47–2.56 GHz (reported max 2.725), memory clock 675–1000 MHz, with hotspot briefly 93 C. No persistent driver override is justified by these measurements. No ROCm profiler executable was installed; the built-in HIP stage profiler was used instead.

## Upstream version check

The checked-out source is `v0.2.21.1` plus the RDNA2 tuning branch; `strata-inkling` reports engine version 0.1.67. Upstream `v0.2.22` is newer, but its release notes explicitly keep AMD/Intel/Apple Inkling at 0.1.67 because those engines did not change. The upstream changes to Inkling engine paths since our base are the engine version bump/build-symbol work and a CUDA peer-card fix; no newer HIP kernel change was found. The 0.2.21 release lists an 8.5 GiB VRAM expert budget workaround for RX 6700 XT-class cards; testing that budget here did not improve the measured rates. See [0.2.21](https://github.com/Niko1221/Strata/releases/tag/v0.2.21) and [0.2.22](https://github.com/Niko1221/Strata/releases/tag/v0.2.22).
