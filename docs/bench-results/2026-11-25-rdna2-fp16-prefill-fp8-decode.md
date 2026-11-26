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


### Decode direct-path trials

At batch one, the runtime uses direct per-route MoE kernels (`B <= 4`); the all-128-expert grouped grid is a prefill path, so compacting its inactive experts would not help decode. The following temporary changes were all reverted:

| Candidate | Decode runs (tok/s) | Numerical check | Result |
|---|---:|---|---|
| QKVRG block: 8 rows / 256 threads → 16 rows / 512 threads | 53.5, 53.9, 53.9 | Same row arithmetic | No measurable change |
| QKVRG: four independent FP32 accumulator chains | 53.2, 53.4, 53.5 | Serial KLD 0.001549, top-1 99.41% | Slower; reject |
| QKVRG: convert BF16 weights and FP32 activations to FP16, use packed `fdot2` | 52.5, 52.8, 52.7 | Serial KLD 0.001549, top-1 99.41%; baseline serial KLD 0.001664, top-1 99.61% | Slightly lower KLD but worse PPL and speed; reject |
| Direct W13: 512-thread blocks, twice the rows | 51.1, 50.0, 52.7 | KLD 0.000000 / top-1 100% on batched reference | Occupancy loss; reject |
| Direct W13: two row-pairs per wave, same 256 threads | 45.7, 45.9, 45.8 | KLD 0.000000 / top-1 100% on batched reference | Serializing work per wave cost 15%; reject |

The serial KLD figures above use one `forward_batch` call per token and FP8 KV, compared against the existing 512-token batched reference. The reference and baseline decode path differ slightly due to FP8 incremental KV updates, so use the baseline serial result as the control. The temporary serial-evaluation hook was removed after the measurements.

On RDNA2, LLVM’s AMDGPU support notes that `v_dot2_f32_bf16` is available starting with gfx11; gfx1031 is gfx10.3, so this direct native BF16 dot path is unavailable here (inference from the target generation). The tested FP16 conversion therefore adds work without a useful speedup. [LLVM AMDGPU change](https://github.com/llvm/llvm-project/commit/aa7eb5723cb4499f35ed1c5455f668ccc078e7c2).


### W2 and post-reduction follow-up (2026-11-26)

The model state width is 2,304. The fused post-W2 reducer therefore launches only 9 blocks at its default 256-thread width. More blocks did not improve end-to-end decode, which points to the W2 expert projections or other per-token stages as the larger costs.

| Candidate | 2K prompt / 256 generated, decode tok/s | Decision |
|---|---:|---|
| Post-W2 reduce/shortconv: 64 threads, 36 blocks instead of 256 threads, 9 blocks | 53.7, 53.8, 53.9 | No measurable gain |
| Direct W2: 128 threads / 4 rows per CTA instead of 256 / 8 rows | 53.4, 53.9, 54.0 | Within run noise; retain 256 |
| One kernel computes all 8 W2 experts, weighted-reduces, and applies shortconv/residual | 37.2, 37.5, 37.5 | Reject; serializing the 8 expert projections loses substantial parallelism |

All candidates were temporary and reverted. The existing parallel per-expert W2 kernels remain the best measured decode route.


### W2 activation staging and dot-chain follow-up (2026-11-26)

A temporary host-side descriptor read during batch-one decode found `inter=1152`; among the observed routed experts, down formats were S2_K (107), S3_K (34), and SF4 (1). The diagnostic was removed afterward. The 1152-wide activation is just beyond the original LDS cache limit of 1024 elements.

| Candidate | 2K / 256 generated decode tok/s | Serial decode KLD (511 positions) | Result |
|---|---:|---:|---|
| Stage up to 1536 activation elements in LDS | 53.4, 53.7, 53.6 | Arithmetic unchanged | No gain vs ~53.9 control |
| Split each four-instruction `fdot2` chain into two independent chains | 52.2, 52.5, 52.5 | 0.001721; top-1 99.61%; PPL 1097.151 | Slower and slightly worse KLD/PPL than baseline serial 0.001664 / 99.61% / 1096.142 |
| Use two accumulators across the S2_K/S3_K subgroup loop | 48.1, 48.4, 48.4 | Not run after clear speed regression | Reject |

These candidates are reverted. Larger shared staging did not help, suggesting the repeated activation loads are already served cheaply enough from cache. The original single-chain dot order remains the fastest tested W2 path.

## Upstream version check

The checked-out source is `v0.2.21.1` plus the RDNA2 tuning branch; `strata-inkling` reports engine version 0.1.67. Upstream `v0.2.22` is newer, but its release notes explicitly keep AMD/Intel/Apple Inkling at 0.1.67 because those engines did not change. The upstream changes to Inkling engine paths since our base are the engine version bump/build-symbol work and a CUDA peer-card fix; no newer HIP kernel change was found. The 0.2.21 release lists an 8.5 GiB VRAM expert budget workaround for RX 6700 XT-class cards; testing that budget here did not improve the measured rates. See [0.2.21](https://github.com/Niko1221/Strata/releases/tag/v0.2.21) and [0.2.22](https://github.com/Niko1221/Strata/releases/tag/v0.2.22).

### W/O fused GEMV row geometry (2026-11-26)

The batch-one W/O projection kernel maps one wave to each output row and stages the activation vector in LDS once per block. I compared the existing 256-thread / 8-row CTA with a temporary 128-thread / 4-row CTA. The LDS copy loop was made block-size-aware so the smaller CTA still covers all activation elements. Per-row arithmetic and accumulation order were unchanged.

| Geometry | 2K prompt / 256 generated, two warmed runs | Decision |
|---|---|---|
| 256 threads / 8 rows (control) | 53.7, 54.0 decode tok/s; 189.9, 190.0 prefill tok/s | Keep |
| 128 threads / 4 rows | 50.1, 50.3 decode tok/s; 189.5, 189.9 prefill tok/s | Reject; about 6.7% slower decode |

The experiment and its block-size-aware LDS copy are reverted. More blocks did not compensate for the smaller CTA on this GPU.

A second W/O geometry trial used 512 threads / 16 rows per CTA. The two warmed decode runs were 55.3 and 52.6 tok/s (mean 53.95), against 53.7 and 54.0 tok/s for the 256-thread control (mean 53.85). The spread is larger than the mean difference, so there is no repeatable gain; the 512-thread change was reverted. The established 256-thread / 8-row geometry remains the production setting.
