# Strata / Inkling-2-Mini SQ-10 RDNA2 tuning notes

Target: Radeon RX 6700 XT (gfx1031), HIP build, Inkling-2-Mini SQ-10. Optimize sustained decode tokens/s while preserving output correctness and stable model serving.

## Current investigation

The active run is checking whether speculative/plain divergence reproduces across three runs at prompt lengths 512 and 2048, plus seed 23. Let it finish before starting any other GPU workload. Record the binary hash, exact command, prompt ID file, seed, and full `[spec-check]` output. A single mismatch is not enough to diagnose a performance regression: compare repeated token streams and first-divergence positions.

Existing counters suggest the fused speculative path can emit multiple tokens when it engages, but engages on only a minority of decode steps. Treat these as separate questions:

1. How much work does an engaged speculative step save after accounting for draft, verification, and tree overhead?
2. Why does the fused path decline or emit zero so often? Inspect eligibility, draft confidence/thresholds, tree construction, and fallback reasons with counters before tuning thresholds.

Do not infer acceptance from aggregate `drafted_sum` / `emitted_sum` without confirming which paths update each counter and how anchor tokens are counted.

## Benchmark protocol

- Keep one GPU benchmark active at a time; avoid competing Strata serving/inference jobs.
- Fix model file, prompt token IDs, seed, context, sampling settings, parallelism, KV mode, expert placement, and binary. Record all of them.
- Warm up once, then collect at least 5 measured runs per candidate. Report median and spread, not the best run.
- Measure prompt processing separately from decode. For decode compare tokens/s at identical prompt and generated-token counts; include TTFT and peak VRAM where relevant.
- Preserve a known-good baseline and rerun it after changes. Change one variable at a time.
- Confirm numerical/token parity with `--spec-check` (including repeated runs and a second seed) before accepting a speculative decoding change.

## Tuning order

1. Establish clean baseline measurements and verify no thermal/power or background-load drift.
2. Profile the decode hot path on gfx1031. Separate launch/dispatch overhead, memory traffic, expert routing, and compute. Use existing ROCm/HIP profiling tools already installed; keep profiling runs distinct from throughput runs.
3. Improve speculative-path engagement only after recording fallback reasons. Sweep one threshold or eligibility condition at a time and report both engaged steps and accepted/emitted tokens per engaged step.
4. For GPU kernels, compare safe variants with identical inputs and correctness checks. Prioritize reducing launches and intermediate reads/writes; check occupancy, LDS use, register pressure, and memory coalescing before increasing tile sizes.
5. Evaluate expert placement and KV settings independently. Record host RAM, VRAM, and steady-state decode throughput; avoid changing placement and kernel settings in the same experiment.

## Result table template

| Change / commit | Build hash | Exact config | Prompt / seed | Runs (median ± spread) | TTFT | Decode tok/s | Spec engaged / steps | Emitted / engaged | Parity | Notes |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|

A change is a TPS win only if the repeatable median improves beyond run-to-run noise, correctness holds, and the target serving configuration remains stable. Document regressions and revert them rather than stacking uncertain changes.

## Hardware and profiler notes

- The RX 6700 XT is RDNA2 `gfx1031` with 12 compute units and 12 GB VRAM. Treat measurements from gfx1030/gfx110x cards as hypotheses, not tuning answers for this GPU.
- Since this card is connected as an eGPU, record the PCIe link width/speed under load and host-to-device traffic when expert placement or offload changes. PCIe transfers can dominate when experts spill to system RAM; compare an all-VRAM run when it fits.
- Use `rocprofv3` first for low-overhead HIP API, copy, and kernel dispatch timelines. Trace only the target kernels when collecting detailed data; profiling adds overhead, so never use traced runs as TPS results. For instruction/occupancy questions, use thread trace/SQTT selectively because it is more intrusive.
- Archive profiler output alongside the run metadata and correlate kernel names/durations with the engines own per-stage timings. Optimize the hottest repeated decode work, not whichever kernel has the largest one-off duration.


Useful primary references:
- AMD GPU architecture table: https://rocm.docs.amd.com/en/docs-6.2.4/reference/gpu-arch-specs.html
- ROCprofiler-SDK overview and tracing guidance: https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/what-is-rocprofiler-sdk.html
- rocprofv3 usage: https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/docs-6.3.1/how-to/using-rocprofv3.html

A relevant community datapoint is the llama.cpp gfx1031 report at https://github.com/ggml-org/llama.cpp/issues/26702. Use it to discover possible gfx1031 pitfalls, but validate any workaround against this engine and exact ROCm stack before adopting it.


## High-upside experiments for this exact setup

The host notes show the biggest measured opportunity is the MoE expert path during prefill (about 62% of a 256-token chunk), while weights already fit in VRAM and the GPU is near full utilization. Start here rather than revisiting power caps or CPU expert placement.

1. **Route-aware MoE weight reuse across the batch.** Instrument selected expert IDs, per-expert token counts, and W1/W2 bytes for real prompts. Then prototype grouping tokens by expert once per layer and reusing each expert tile across its routed tokens, with a small-token fallback. The upside is reducing repeated weight reads and launch/dispatch overhead; benchmark M=1 decode separately from prefill because the best schedule may differ.
2. **Fuse the MoE epilogue.** Test fusing W1 activation/gating with W2 accumulation or combining dispatch/indirection with the first projection, while keeping a simple unfused reference. Aim to eliminate intermediate writes and launches; compare end-to-end prefill and decode, not just isolated GEMM.
3. **Narrow speculative decoding to code workloads.** Existing notes show DSpark was slower on chat but faster on code. Instead of forcing it globally, build a prompt-class or online break-even gate: enable speculation only when recent accepted tokens per draft and measured draft/verify cost beat plain decode. Do not change sampling semantics or force speculation before parity checks.
4. **Test KV precision/attention as context grows.** At 8k-24k, decode falls substantially while KV grows. Compare current FP8 against the supported lower-memory KV format (if numerical quality is acceptable) and profile attention at 8k, 16k, 24k. Separate KV bandwidth savings from cache-fit effects; report quality and throughput together.

The most promising risky kernel direction is an expert-grouped, split-K or persistent schedule tuned for the actual routed-token histogram, not simply a larger tile. Prior register-tile attempt regressed, so measure register use/occupancy and retain the current kernel as fallback. Any int8 activation path that is not bit-identical needs an explicit quality/speed decision and full-generation comparison before deployment.

## Verified 2026-11-25: FP16 prefill, FP8 decode cache (short context)

An experimental benchmark-only path now runs prompt prefill with FP16 KV, then converts resident KV pages to FP8 E4M3 before generation. The conversion time is included in decode timing. On the RX 6700 XT with Inkling-2-Mini-SQ-10, 2,048 prompt / 256 generated tokens, parallel=1, all experts in VRAM (8.77 GiB), no RAM experts:

| KV path | Runs | Prefill tok/s | Decode tok/s |
|---|---:|---:|---:|
| FP8 throughout | 186.4, 186.8, 186.9 | median 186.8 | median 53.6 |
| FP16 prefill -> FP8 decode | 208.2, 208.3, 207.9 | median 208.2 | median 56.9 |

Improvement: +11.5% prefill and +6.2% decode. Teacher-forced parity over the post-conversion portion of the 512-token reference: 255 positions, mean KLD 0.001667, top-1 agreement 100%. This is an experimental `STRATA_BENCH_FP8_DECODE=1` path, not enabled in serving. It refuses host-offloaded pages and UMA/multi-GPU configurations. Long contexts may not fit the temporary conversion buffers; the 8K FP16 case spilled KV to host and is not a supported win. Validate memory headroom and compare full generations before promoting to serving.
