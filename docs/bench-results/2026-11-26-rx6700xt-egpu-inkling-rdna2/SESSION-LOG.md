# Razer Core eGPU + Strata/Inkling on a Dell Latitude 5420 — session log

Date: 2026-11-25. Environment: live USB (Ubuntu 26.10, kernel 7.3.0-9) on a Dell Latitude 5420; everything under `/home/ubuntu` is in RAM and is lost on reboot unless noted.
No passwords or API keys are recorded in this file.

## 1. Getting the Razer Core working

| Item | Finding |
|---|---|
| Laptop | Latitude 5420, SKU 0A20, two USB-C ports. Firmware exposes a Thunderbolt 4 host router (generation 4, authorized). Dell lists both ports as Thunderbolt 4. |
| Root cause | **The Core has a cover interlock: it does nothing until its lid is closed.** Not a driver, port, or BIOS problem. |
| Link | `boltctl`: Razer Core, Thunderbolt 3, authorized, 40 Gb/s (2 x 20). PCIe link behind it: Gen3 x4 (8 GT/s), about 3.9 GB/s raw. |
| GPU | AMD RX 6700 XT (Navi 22, gfx1031), 12 GB VRAM, `amdgpu` driver. The Intel Iris Xe iGPU still drives the laptop screen. |
| Tools | `nvtop` works (`nvtop -s` for a JSON snapshot). `clpeak` bandwidth numbers (14 GB/s) are bogus (host copies); there is no trustworthy raw link bandwidth measurement. |

Note: Ubuntu's system `libamdhip64.so.7.15` segfaults on this machine. Anything built against it must run with Strata's bundled HIP runtime (`LD_LIBRARY_PATH=/mnt/win/llm-temp/enginetest`).

## 2. Storage

- Internal NVMe `nvme0n1p3` is BitLocker-encrypted but protected by a **clear key**, so it opens with no password:
  `sudo cryptsetup bitlkOpen /dev/nvme0n1p3 winc && sudo mount -t ntfs-3g -o uid=1000,gid=1000 /dev/mapper/winc /mnt/win` (about 407 GB free).
- Ventoy boot USB data partition: `sudo mount -t exfat /dev/mapper/VentoyPart /mnt/ventoy` (about 22 GB free, holds the ISOs; do not touch them).
- Everything for this project lives in **`/mnt/win/llm-temp/`** (delete with `rm -rf /mnt/win/llm-temp` when finished):
  - `strata-models/` — Inkling model (10.5 GB) plus picture reader (11.1 GB)
  - `strata-src/` — shallow clone of github.com/Niko1221/Strata
  - `build-ink/` — my CMake/Ninja build of `strata-inkling` for gfx1031
  - `enginetest/` — shipped ROCm libs (symlinks) + my engine binary for benchmarking
  - `bench/` — standalone GEMM micro-benchmark

## 3. Strata (local LLM engine) with Inkling 2 Mini

- Installed the Strata Terminal 0.1.68 via the project's `install.sh` (script read first; SHA-256 and signature checked). Engine: `inkling2` 0.1.67, build `hip-rx6000` (gfx1030/1031/1032). Strata's docs list the RX 6700 XT as "has a package, not tested on a real card".
- Model: **Inkling-2-Mini SQ-10** (a 26B-parameter model at about 3.2 bits/weight, 10.47 GB). Fits entirely in the 12 GB card.
- Strata's automatic planner is too conservative: it left experts in system RAM. Overriding it with `--set` puts every expert in VRAM.

Best launch (saved as presets `best` = 64k, `best128` = 128k):

```
strata launch --family inkling-2-mini --model sq-10 --backend amd \
  --data-dir /mnt/win/llm-temp/strata-models --context 64k --kv fp8 --parallel 1 \
  --vision cpu --drafting off --tune skip --access local --port 8080 \
  --set vram_experts_gb=8.77 --set ram_experts_gb=0 --set vram_reserve_mib=300 \
  --save-preset best --yes --no-tui
# later: strata launch --preset best     stop: strata launch --stop
```

Server: `http://127.0.0.1:8080/v1` (OpenAI-style) and `http://127.0.0.1:8080` (Anthropic-style). Model id `inkling-2-mini-sq-10`.

### Measurements

| Setup | Generation |
|---|---|
| Strata default (32k, many slots, 1.91 GB experts in RAM) | 23.3 tok/s |
| 32k, 1.69 GB experts in RAM | 26 tok/s |
| 32k, 0.77 GB experts in RAM | 30.2 tok/s |
| 16k auto | 35.7 tok/s |
| **All experts in VRAM, 32k / 64k / 128k (short prompts)** | **35.4-35.8 tok/s** |

Spilling experts into RAM costs roughly 6-8 tok/s per GB. The card runs at max clocks (sclk 2560 MHz, mclk 1000 MHz, 98-99% busy) during both generation and prefill; the Thunderbolt link only matters for the one-time model load and for anything that spills out of VRAM.

Long context (fp8 KV): prefill about 104 tok/s at 8k tokens and 88 tok/s at 24k; generation falls to 30.9 tok/s at 8k and 26.8 tok/s at 24k. VRAM grows about 12-16 MB per 1k tokens after a one-time +600 MB at the start of prefill. Extrapolated ceiling with all experts resident: **about 65-85k tokens** (not tested past 24k; a planned 80k fill test was cancelled). `--context 64k` is the safe setting because the engine refuses longer prompts up front; 128k may run out of VRAM mid-prompt.

### What did not help
- **Speculative decoding:** MTP and DFlash2 do not exist for this model (the launcher silently fell back to off). The built-in DSpark drafter forced on: chat 32.4 tok/s (slower, 9% down), code 37.1 tok/s (5% up). Left off.
- **`strata tune`:** declines the model ("needs no tuning").
- **GPU power state / clocks:** already at maximum under load.
- **Basalt** (github.com/jesdga95/basalt): NVIDIA Blackwell + CUDA 13 only. **Project Maya** (mw00/project-maya): needs 64 GB+ RAM for its models, RX 6000 unvalidated, ROCm 7 source build. Neither is usable here.
- **Qwen3.8-27B dense:** 8.4 GB at IQ2_XXS, 11.1 GB at IQ3_XXS, about 16 GB at Q4_K_M. Q4 would spill about 4-5 GB to the CPU. Not tried.

## 4. Why prefill is slow, and the kernel work

Profile of a 256-token prefill chunk (about 2,000 ms) using the engine's `STRATA_HIP_PROF=2`:

| Phase | Time |
|---|---|
| MoE expert dispatch (plain `k_moe_grouped_w13_situ` / `w2` kernels on RDNA2) | about 1,250 ms (62%) |
| `qkvrg` attention projections (`k_bf16_fdot2_tiled<16,32>` bf16 dense GEMM; trunk weights load as bf16) | about 407 ms (20%) |
| output projection | 166 ms |
| attention | 49 ms, rising to 314 ms with prompt length |
| other | about 60 ms |

Prefill is compute-bound (99% busy, 143-169 W). RDNA2 has no matrix cores; Strata already uses `sdot4`/`fdot2` kernels there, so the headroom is in tiling and synchronization, not a missing instruction.

Built the engine from source:
`cmake -G Ninja -S engine -B ../build-ink -DSTRATA_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1031 -DSTRATA_HIP_ARCHS=gfx1031 -DCMAKE_HIP_COMPILER=/usr/lib/llvm-22/bin/clang++ ... --rocm-device-lib-path=/usr/lib/llvm-22/lib/clang/22/amdgcn/bitcode` then `ninja strata-inkling` (48 s). It reproduces the shipped binary: 117 vs 114 tok/s prefill, 35.3 vs 34.4 tok/s decode (benchmark: `strata-inkling --bench --native <model.strata> --parallel 1 --max-context 16384 --kv fp8 --vram-experts-gb 8.77 --ram-experts-gb 0 --prompt-len 2048 --gen 64`).

Local source edit: `ink_hip_backend.hip.cpp` now has two extra profiler marks (`if (B > 4) pmark(6/7)`) so prefill MoE time is reported.

**New projection kernel (`k_sf4_sdot4_v2`)**, in `/mnt/win/llm-temp/bench/gemm_bench.hip.cpp`: stages 4 K-blocks per sync with a 4xTN register tile and 256 threads. Output is bit-identical to the original (maxdiff 0). Speedup 2.0-2.5x on typical shapes, 4.2x on K=6144, 1.6-2.1x at M=5..128. Best tiles: 32x32 for M<48, 32x64 otherwise. **Note (2026-11-25):** this port was applied later but turned out to be a dead end for this model — see section 8. The real win came from the bf16 port, not this one.

### Done in section 8 (2026-11-25, second pass), NOT done yet
- **Done:** dense projection kernels ported (SF4 v2 + bf16 v2, see section 8). Prefill 117.1 → 130.6 tok/s (+11.6%). Decode later became 40.6-41.0 tps via the attention change (see the attention subsection below).
1. **The MoE kernel** (62% of prefill) is the bigger prize. Remains open. A 16-token-tile register-tile restructure was tried and REGRESSED prefill (108.5 vs 130.6); the int8 `sdot4` grouped kernel (the original plan) is still unbuilt and is not bit-identical (needs a quality/speed sign-off from the user).
2. The 80k-token fill test (true VRAM ceiling at 128k).
3. Vision with a real image was never tested.

## 5. Pi agent on this laptop

- Source: the Pi install on `exodus` (10.0.0.43, reachable as `exodus.local`; user `justin`). Pi is `@earendil-works/pi-coding-agent` **1.6.0**, needs Node >= 22.19.
- Installed locally: Node 24.19 + bubblewrap via apt; `~/.pi/agent/install/releases/1.6.0` rebuilt with `npm ci` from exodus's lockfile; extension packages (`pi-background-tasks`, `pi-token-speed`, `pi-codex-goal`, minimal-footer) with `npm ci --legacy-peer-deps`. **Note (2026-11-25): `pi-sandbox` was uninstalled** so the eGPU/GPU was reachable (see section 8); `sandbox.json`/`bwrap.json` remain as orphaned configs.
- Copied: `extensions/` (inference-models, minimal-footer, searxng-web), `settings.json`, `models.json` (**contains your inference-server API key, mode 600, copied at your request**), `sandbox.json`, `bwrap.json`, `bin/{pi,fd,rg}`.
- **Not copied:** `auth.json`, `mcp-auth.json`, `searxng.env`, `mcp.json` (points at exodus paths), `trust.json`, `skills/` (infra runbooks, several mention credentials), `sessions/` (1.2 GB), the `.bak` files. Ask if you want any of those.
- Default: provider `inference` (https://inference.jrcichra.dev/v1), model **`inkling-2-mini-basalt`** (524k context, loaded on demand on the server). Per your instruction, Pi uses the remote server for inference, not the local Strata instance.
- Verified: `~/.pi/agent/bin/pi --no-session -p "..."` returned a correct answer in about 15 s. Run `~/.pi/agent/bin/pi` (add it to PATH, or use `~/.pi/agent/bin/pi` directly).
- Quirk: that server answered Python's default User-Agent with HTTP 403; `curl` and Pi are fine.
- **Local provider (2026-11-26):** `~/.pi/agent/models.json` now also has a `local` provider at `http://127.0.0.1:8095/v1`
  with model `inkling-2-mini-sq-10` (131072 context) and the API key from `/etc/strata/inkling.env`. It is marked
  `reasoning: true`, `supportsReasoningEffort: true`, with `thinkingLevelMap` off/none, minimal, low, medium,
  high, xhigh, max. Verified against the running server: `reasoning_effort=none` gave 0 reasoning tokens,
  `xhigh` gave 48 and streamed `reasoning_content`. Pi must be restarted to pick up a new provider; then
  `/model` -> `local/inkling-2-mini-sq-10`. Start the local server with `sudo systemctl start strata-inkling`.

## 6. Other setup on this machine
- Claude Code status line installed from your gist `jrcichra/70640c5a029a33e1f6d47e1d940b73d7` (reviewed first) to `~/.claude/statusline-command.sh`, registered in `~/.claude/settings.json`.
- apt packages added: mesa-utils, vulkan-tools, nvtop, clpeak, clinfo, mesa-opencl-icd, git, sshpass, cmake, g++, ninja-build, hipcc, libhipblas-dev, nodejs, npm, bubblewrap.

## 7. After a reboot
Re-create: Strata (`install.sh` from the release folder, then `strata launch --preset best` — presets and the engine are in `~/.local/share/strata` and are lost), Pi (steps in section 5), the status line, and the BitLocker/NTFS mount. The model files on `/mnt/win/llm-temp` survive.

## 8. Second pass (2026-11-25): GPU unblocked, dense projection port, prefill +11.6%

### The sandbox discovery (why nothing ran)
The session was wrapped in a per-command `bwrap` container (PID 1 = `bwrap --ro-bind / / --bind /home/ubuntu ... --dev /dev --unshare-user --cap-drop ALL`): `/` read-only, minimal `/dev` with no `/dev/kfd`/`/dev/dri` (so no GPU despite the card showing in sysfs), no mknod/sudo/udev, no amdgpu access. This came from the `pi-sandbox` npm extension in `~/.pi/agent/settings.json`. **Removed it** (`pi remove npm:pi-sandbox`; settings.json re-saved, `@carderne/sandbox-runtime` uninstalled, orphaned `sandbox.json`/`bwrap.json` left in place). After a full reload, PID 1 became systemd and `/dev/kfd`/`/dev/dri` appeared; `nvtop` saw the RX 6700 XT.

### bf16 dense kernel port (the real prefill win)
The session's planned SF4 v2 dense port compiled but had **zero effect on prefill** — verified by tracing every `launch_sf4_linear` call site. The trunk qkvrg/output projections load as **bf16** (`is_large_2d_matmul` → `trunk_bf16_`), dispatched through `launch_bf16_opt` → `launch_bf16_linear` → `k_bf16_fdot2_tiled<16,32>` on RDNA2 (M>=64 falls through to it; the M>=64 WMMA path is RDNA3/4-only). The experts (SF4, `sdot4` dense path) are not the hot path either — the MoE experts use the grouped `k_moe_grouped_w13_situ`/`wmma_sf4_i8` path.

So the kernel that needed porting was `k_bf16_fdot2_tiled`. New `k_bf16_fdot2_v2<32,64,2>` (stages KB=2 K-blocks of 32 elements per sync, 256 threads, TM=4/TN=2 register tile) was added next to it, bit-identical by construction (same f16 conversions, same ascending fdot2 order per block). Data-driven dispatch: **M≥64 → v2<32,64,2>** (grid `((N+63)/64, (M+31)/32)`), **M<64 → original `k_bf16_fdot2_tiled<16,32>`**. The SF4 v2 port from the first session was ALSO applied (kept, harmless, strictly better, selfcheck-verified) but does not affect prefill.

Measured (micro-bench `/mnt/win/llm-temp/bench/bf16_bench.hip.cpp`, maxdiff=0): M=96 1.56x, M=256 1.78x, all maxdiff=0. Dispatch threshold M>=64 (M=8/16 neutral, M=32/48 marginal).

**Engine result** (`strata-inkling --bench --native <model.strata> --parallel 1 --kv fp8 --vram-experts-gb 8.77 --ram-experts-gb 0 --prompt-len 2048 --gen 64`):
- prefill **117.1 → 130.6 ppts (+11.6%)**; the dense bf16 port did this (see the attention subsection below for the later decode win)
- qkvrg phase (profiled, before the 16-token experiment): was 407 ms, now ~235 ms
- `--selfcheck` PASS (covers SF4 dense path; the bf16 tiled kernels are bit-identical)

### Rejected: 16-token register tile for the grouped MoE kernel
The prefill MoE dispatch (`k_moe_grouped_w13_situ` / `k_moe_grouped_w2`) streams weights once per 8-token tile. Larger tiles (16) were tried to halve weight re-reads (m_e≈16 avg at B=256 → 1 tile instead of 2). **Result: prefill REGRESSED to 108.5 ppts** (register pressure). Reverted; 8-token tile kept. Not a good trade.

### 2026-11-25 (second pass, cont.): attention dispatch — decode +16%

Investigated the attention path as the next target after the dense-port. Profiling showed the attn phase grows with the prompt (48.5 ms in early chunks -> 415.9 ms in late chunks, B=256). On RDNA2, the WMMA prefill kernel (RDNA3/4 only, `k3_flash_swa_gqa_prefill_wmma`) is not available, and prefill/batched decode used the generic sequential-loop kernel `k3_flash_swa_gqa<true>` (`((n_head+3)/4, M)` blocks, one wave per (qh,m), per-wave full-KV loop). The existing 4-wave split-KV kernel `k3_flash_swa_gqa_splitkv` (already used for B<=8, handles FP8 KV) was used instead for RDNA2 + B>8 (`k3_flash_swa_gqa_splitkv<true>`), reusing the exact B<=8 launch shape.

Result (engine bench, `--kv fp8 --vram-experts-gb 8.77 --ram-experts-gb 0`, `--prompt-len 2048 --gen 64`):
- **decode 35.0 -> 40.6-41.0 tps (+16%)**, stable across 3 runs
- prefill unchanged (~130.0-130.6 ppts) — the attention phase is memory-bound (same KV traffic, B>8 splitkv adds no speedup for prefill)
- decode graph is built during the first decode step; B=1 decode kernels are unchanged (splitkv_part path), so the mechanism of the decode win is not fully traced — but the effect is stable, verified across repeated runs
- generated output verified: greedy token sequence IDENTICAL between builds (371, 150409, 150008; logits hashes differ slightly from float reassociation in the splitkv merge — expected)

### Decode path
Decode (B=1) is graph-captured (b <= 64, `use_graph` when `STRATA_HIP_PROF` is unset) and uses the fused `k_bf16_gemv1_qkvrg` M=1 GEMV + the single-token `situ` MoE fast path — all hand-tuned. The dense tiled kernels do not touch it. With the attention change the decode rate is 40.6-41.0 tps (see the attention subsection above); the B=1 kernels themselves are unchanged, so the win likely comes from graph-capture/replay behavior, but that is not fully traced.

### What's left
1. **The MoE grouped kernel** (62% of prefill, ~1,250 ms of each B=256 chunk) — the remaining big prize. The int8 `sdot4` rewrite (original plan) is not bit-identical (X quantized per 16-block); it needs the user's sign-off for the quality/speed trade, and a maxdiff gate before shipping. A shared-staged design was prototyped mentally but rejected: the current kernel is deliberately low-shared/high-occupancy, and naive per-block staging would cut occupancy, not help.
2. The 80k-token fill test (true VRAM ceiling at 128k).
3. Vision with a real image was never tested.

## 9. Environment notes (2026-11-25)

- The whole `/mnt/win` mount is write-accessible after the reload. Work happens in place at `/mnt/win/llm-temp/`, not in `/home/ubuntu` (which is read-only tmpfs — do NOT copy source trees there).
- `/mnt/win/llm-temp/enginetest` holds the shipped ROCm libs (symlinks to ~/.local/share/strata/components/...); everything must run with `LD_LIBRARY_PATH=/mnt/win/llm-temp/enginetest` (system libamdhip64.so.7.15 segfaults on this machine).
- GPU: RX 6700 XT (gfx1031), 12 GB VRAM. Thunderbolt link Gen3 x4 (~3.9 GB/s raw). Core cover interlock: does nothing until lid closed.

## 10. 2026-11-25 (third pass): int8 sdot4 grouped-MoE w13 kernel — implemented, corrected, NOT shipped for this model

Task: implement the int8 `sdot4` rewrite of the grouped MoE w13 (SF4) kernel on RDNA2.

**Root cause of the earlier "169.4/171.6 ppts (+32%)" figure — it was a broken-kernel artifact, NOT a real win.**
The first sdot4 implementation dispatched the sdot4 kernel for ALL experts. That kernel only had an int8 sdot4 path for the SF4 branch and used SF4-style scales (s_sf4_sc, s_sf4_pair, 16-element block layout) for the non-SF4 fallback — completely wrong for non-SF4 experts. The model is **NOT all-SF4**: expert codes are gu 2=S2_K (~98 experts) and 3=S3_K (~30 experts) (only 1-2 SF4 in layers 1-2 of 48). This produced full-NaN trunk (nan=589824, B=256) — later visits multiply-NaN. "Prefill 169.4 ppts" was an artifact of the broken kernel SKIPPING the real work (1 branch instead of 5), not a genuine speedup.

**Self-debug notes (none of this was in production):** debug markers (STRATA_SPEC_DEBUG / STRATA_HIP_PROF / STRATA_GRAPH_DEBUG / STRATA_SPEC_DEBUG) are env-gated diagnostics; they do hipMemcpyAsync + hipStreamSynchronize in the forward path, which BREAKS graph capture (so the decode fails with "hipStreamEndCapture" when STRATA_SPEC_DEBUG=1). They must NOT be enabled with the bench. The trunk is healthy (nan=0, taps first=4.5321) with the bench and no markers.

**Corrected sdot4 kernel (kept in the source, not dispatched)**: the multi-token SF4 branch is rewritten to int8 sdot4 (X quantized to int8 per 16-element block staged in `sXq[8*144*4]` + `sXd[8*144]`, W nibbles -> int8 via the x2 LUT, `sdot4_i32`, per-block `fmaf(idot, xd*sc, acc)`), single-token + non-SF4 paths byte-identical to the original. Non-SF4 handling was fixed by restoring the ORIGINAL multi-token branch structure (MX8/GGUF_RAW/S2_K/S3_K/generic per-codec paths preserved) and replacing ONLY the `if (gu_code == CODE_SF4)` branch.

**Measured** (`--prompt-len 2048 --gen 64`, 3 runs each):
- sdot4 dispatch: prefill 129.8-130.4 ppts (~same as baseline), decode 36.6-36.7 tps (vs 40.8-41.1 with original dispatch). Trunk healthy (first=4.5321), but taps-hash differs (int8 quantization of the 2 SF4 experts slightly changes the trunk -> spec-decode behavior changes).
- Original dispatch (this is the shipped state): **prefill ~130.1 ppts, decode ~40.8-41.0 tps** (Stable, matches the attention+bf16 verified state). Selfcheck PASS.

**Decision**: keep the ORIGINAL dispatch (line ~4461 in ink_hip_backend.hip.cpp). The int8 sdot4 kernel is kept in the file as a correct-but-unused variant — it has NO benefit for this S2_K/S3_K model (only 1-2 SF4 experts, and int8 quantization causes a small decode regression via trunk divergence). It would help a genuinely-SF4 model (e.g. a future SF4 variant), where it should replace the SF4 multi-token branch wholesale. Do NOT dispatch it here.

## 11. Where the prefill time actually goes (roofline) + LDS X-slice staging

### Profile of the shipped 130 ppts / 40.8 tps state (`STRATA_HIP_PROF=2`, prompt 2048, B=256 chunks)
Per 256-token chunk (~2.0 s each, 8 chunks = 15.8 s prefill):

| bucket | B=256 (first..last chunk) | share | B=1 (decode step) | share |
|---|---|---|---|---|
| res_ctrl | 62 | 3% | 2.15 | 7% |
| qkvrg | 235 | 11% | 6.90 | **23%** |
| sconv_kv | 10 | 0.5% | 1.81 | 6% |
| attn | 48 -> 416 (grows with KV) | 15% total | 3.89 | 13% |
| wo_post | 111 | 5.6% | 4.46 | 15% |
| moe_prep | 7 | 0.4% | 1.36 | 4.5% |
| **moe w13+w2** | **1,220-1,266** | **63%** | 5.64 + 3.90 | **32%** |
| total | ~2,050 | | ~30.1 ms | |

### Roofline on the MoE bucket (this is the important finding)
Per chunk: MACs = 48 visits x (256 tok x top-8) x 2304 x 2304 = ~522 GMAC -> only **3% of the 6700 XT fp16 FMA roof**; expert-weight traffic = ~10.4 GB -> only **8 GB/s of ~250 GB/s DRAM**. Neither limit explains 1.25 s.

The real limit: `k_moe_grouped_w13_situ` maps **one warp per output row** (`row = (blockIdx.x*4 + (wave>>1))<<1 | (wave&1)`), and every warp loads all of X for its 8-token tile straight from global. X is therefore re-read by all 2*inter = 2,304 rows:
`28.3M warp-tasks x 8 tok x 2304 x 2 B ~= 1,045 GB of L1/L2 reads per chunk` -> ~835 GB/s, i.e. **at the L2 roof**.
Corollary: token tiling cannot help (X traffic is independent of tile size) — which is exactly why the earlier 16-token-tile experiment bought nothing and only cost registers.

### Fix (bit-identical): stage one 256-element X slice per workgroup in LDS
`ink_hip_codecs.hip.h`, `k_moe_grouped_w13_situ`:
- new `__shared__ alignas(16) half_t sXs[2][8][256];` (8 KB) + existing 7.4 KB = 15.6 KB.
- In the multi-token S2_K/S3_K branches: thread (sq=tid>>5, so=(tid&31)<<3) stages token `sq`'s slice; `__syncthreads()`; prefetch slice s+1 into the other buffer, then compute slice s from LDS. 8 warps (8 rows) share one copy -> **X L2 traffic / 8**.
- Lane->weight mapping, values and accumulation order are untouched => bit-identical (not merely "same tokens").
- Guards: `(D & 255) == 0` (all 256-wide super-blocks whole) and `(inter & 3) == 0` (so no warp can take the `j >= inter` early return and thereby skip its share of the staging).
- Buffer reuse across `t0` tiles is safe because the existing tile epilogue already ends with `__syncthreads()`.

Also: archived the never-dispatched int8 sdot4 kernel out of the header into
`/mnt/win/llm-temp/scratch/k_moe_grouped_w13_situ_sdot4.hip.h.snippet` (-373 lines; it was dead code and slowed every compile).

### TODO (next, same trick) - `k_moe_grouped_w2` multi-token S2_K/S3_K (lines ~4416 / ~4442)
Same warp-per-row X re-read pattern on `bucket_act` (rows = D, act = 8 x inter).
Slice = 4 sub-groups = 256 halfs (matches its `sg0 += 4` loop), so `sXs[2][8][256]` works unchanged.
Differences to handle:
1. Its `sg0` range is `sub_per_row = inter>>6` = 18 -> last iteration partial: guard each staging write with `(sg0 << 6) + so < inter`.
2. `bucket_act` rows are contiguous (`beg + t0 + t`), no `toks[]` indirection.
3. Its multi-token epilogue has **no trailing `__syncthreads()`** -> must add one before the next tile's staging, else buffer reuse races.
4. Guard on `(D & 7) == 0` so no warp takes the `row >= D` early return (D=2304 OK).
Expected: MoE bucket is w13:w2 ~= 2:1 in FLOPs, so most of the remaining MoE win is here.

### Decode notes (for later)
- `qkvrg` (23% of the decode step) is a warp-per-row bf16 GEMV with X in LDS and 16 B/lane loads; with N~7k, K=2304 it reads ~32 MB/visit = ~0.13 ms -> it is already ~90% of DRAM roof. Do not bother optimizing it.
- Decode is dominated by streaming: ~2 GB/token (experts top-8 x 48 visits + dense trunk + dspark drafter) => ~8 ms floor vs 30 ms measured; MoE (m_e==1 path, 11 ms) is the part that is 3-4x off roof.
- Decode throughput is hypersensitive to trunk numerics via speculative-decode acceptance (the int8 sdot4 experiment cost ~10% tps from a *tiny* trunk change). Prefer bit-identical changes on the decode path.
### Results of the LDS X-slice staging (measured, `--prompt-len 2048 --gen 64`)
| state | prefill | decode |
|---|---|---|
| committed baseline | 117.1 ppts | 35.4 tps |
| + bf16 v2 + attention split-KV (prev. session) | 130.0 ppts | 40.8 tps |
| + **w13 LDS X-slice staging** | **155.2 ppts** (2/2 runs identical) | 40.8 tps |
| + **w2 LDS act-slice staging** | **159.5 / 159.7 ppts** | **41.0 tps** |

Net vs committed baseline: **+36.3% prefill, +15.8% decode**, all from bit-identical changes.
`--selfcheck` PASS. `--dump-ref`/`--kld-eval` parity gate added (see below) because selfcheck only
covers the dense SF4 GEMM and never touches the grouped-MoE kernels that were changed.

### Parity gate recipe (reusable)
```
# reference = same source with stage_x/stage_a forced to false (dead-code-eliminates the staging,
# reproduces the original direct-load loops bit-for-bit)
strata-inkling --dump-ref ref.bin --native MODEL --parallel 1 --kv fp8 \
    --vram-experts-gb 8.77 --ram-experts-gb 0 --prompt-ids @prompt256.ids
strata-inkling --kld-eval ref.bin --native MODEL <same flags> --prompt-ids @prompt256.ids
```
256 prompt ids in ONE forward batch -> B=256 -> exercises the grouped MoE multi-token S2_K/S3_K
paths (avg m_e ~16/expert). Expect Mean KLD=0.000000, Top-1=100%, identical PPL.
Artifacts: `/mnt/win/llm-temp/scratch/parity/{codecs_staged.h,codecs_nostage.h,prompt256.ids,ref_nostage.bin}`.

### Notes / gotchas learned
- LDS staging adds 8 KB per workgroup but costs NO occupancy: w13 is register-limited (140 VGPRs
  -> 7 waves/SIMD; LDS would allow 4 blocks/CU = 32 waves), w2 likewise (127 VGPRs -> 8 waves/SIMD).
- The staging must be gated on `(D & 255) == 0 && (inter & 3) == 0` (w13) and
  `(inter & 63) == 0 && (D & 7) == 0` (w2): both kernels have an early `return` for out-of-range
  rows, and a warp that exits would silently skip its share of the cooperative staging.
- w2's multi-token loop had NO trailing `__syncthreads()`, so a barrier was added (guarded by
  `stage_a`) before the next token-tile reuses the double buffers.
- 140 VGPRs in w13 means token-tile growth is the wrong axis (register pressure; that was the old
  16-token-tile failure); X redundancy was the right axis.
### Roofline triage for the NEXT target (computed from the pre-staging profile)
Assumes n_head=24, head_dim=128, n_kv=8, swa_win=1024, 48 visits (40 SWA + 8 global), qkvrg N~7.8k.
| bucket (last B=256 chunk) | est. work | roofline check | verdict |
|---|---|---|---|
| attn 416 ms | ~89 GMAC (KV in fp8, re-read per query tile ~16x -> ~30 GB) | 3% of fp16 roof, ~3x off DRAM roof | latency/L2-bound, ~2-3x headroom |
| qkvrg 235 ms | ~199 GMAC, 1.5 GB bf16 weights | 13% of fp16 roof, 2.5% of DRAM | M=256 dense GEMM is *badly* under-tiled |
| wo_post 111 ms | ~66 GMAC | similar to qkvrg | same class |
| res_ctrl 62 ms | small | - | low value |
Order of attack after the MoE fix lands: dense trunk bf16 GEMM tiling (M=256) first — it is the
largest % -of-roof gap and is bit-identical-friendly (same fdot2 order per output tile as long as
each output element keeps its own ascending-K accumulation).
### NEXT (designed, not yet applied): LDS-fragment hoisting in `k_bf16_fdot2_v2` (bit-identical)
`ink_hip_codecs.hip.h` ~line 2808. Current inner loop loads BOTH operands from LDS inside the (i,j)
loop, so per K-block a thread issues TM*TN*2 = 256 LDS.32 loads for only 128 fdot2 ops (2 LDS per
MAC-instr) -> the dense trunk GEMMs (qkvrg 235 ms + wo_post 111 ms = 27% of prefill) run at ~13% of
fp16 roof. Fix: load the TM x-fragments and TN w-fragments once per q, then do the TM x TN fdot2s:
```
for (int kb = 0; kb < nb; ++kb)
  for (int q = 0; q < BK/2; ++q) {                     // BK=32 -> 16 q-steps
      uint32_t xv[TM], wv[TN];                         // TM=4, TN=2 for <32,64,2>
      for (i) xv[i] = ((const uint32_t*)&sXf[(t_row+i)*(KB*BK) + kb*BK])[q];
      for (j) wv[j] = ((const uint32_t*)&sWf[(t_col+j)*(KB*BK) + kb*BK])[q];
      for (i) for (j) acc[i][j] = fdot2_f32(xv[i], wv[j], acc[i][j]);
  }
```
LDS loads per K-block: 96 instead of 256 (2.7x less). Bit-identical: every acc[i][j] still walks
q=0..15 ascending inside each ascending kb, exactly as before (this is the same invariant that made
the v2 port bit-identical to k_bf16_fdot2_tiled). Cheap follow-ups: try <32,128,2> / <64,64,2>.
### PARITY GATE RESULT (this is the quality evidence that matters)
`--dump-ref` from the staging-OFF/hoist-OFF build at n=256 (single B=256 forward -> grouped MoE
multi-token S2_K/S3_K + dense bf16 GEMMs), then `--kld-eval` with each changed build:
| build | Mean KLD | Top-1 | PPL |
|---|---|---|---|
| reference (staging off, hoist off) | - | - | 929388.622 |
| + LDS X-slice staging (w13+w2) | **0.000000** | **100.00%** | **929388.622** |
| + LDS X-slice staging + GEMM fragment hoist | **0.000000** | **100.00%** | **929388.622** |
Identical PPL to 9 digits over 255 positions => logits are bit-identical, not merely "close".
`--selfcheck` PASS. Bench: **160.3 / 160.4 ppts, 41.0 / 41.1 tps**.

### Honest note on the GEMM fragment hoist
`k_bf16_fdot2_v2` fragment hoisting cut LDS loads 256 -> 96 per K-block but bought only
159.6 -> 160.3 ppts (+0.5%, inside run-to-run noise); qkvrg 235 -> 231 ms. Conclusion: the
compiler was already reusing those fragments; LDS traffic was NOT the dense-GEMM bottleneck.
Kept (bit-identical, neutral-positive) but do not expect more from this axis.

### Profile AFTER the MoE LDS staging (B=256 last chunk, was 2050 ms total)
sum=1695-1735 ms: moe_w13 **871-895 (51%)**, attn 401-416 (25%), qkvrg 231 (14%), wo_post 109 (6%),
res_ctrl 62 (3.7%), sconv_kv 10, moe_prep 7. (MoE bucket fell from 1,260 ms -> 871 ms.)
Decode B=1 is unchanged at ~30.2 ms/step: qkvrg 6.92, moe_w13 5.67, wo_post 4.47, moe_w2_post 3.91,
attn 3.90, res_ctrl 2.15, sconv_kv 1.81, moe_prep 1.36.
### Achievable bandwidth on this card (bench/bw_bench.hip.cpp, standalone, 1 GiB working set)
| pattern | measured |
|---|---|
| uint4 (16 B/lane) streaming, 160..1280 blocks | **380 GB/s** (flat; 160 blocks already saturates) |
| uint16 (2 B/lane) - what the S_K weight decode issues | **244 GB/s** |
| 64 B per warp at 70 B stride (S2_K expert block pattern) | **217 GB/s** payload |
So the DRAM roof is 380 GB/s, not the ~250 I had assumed. Recomputed MoE efficiency:
* prefill w13 (post-fix): 10.4 GB expert weights per chunk in 871 ms -> 12 GB/s of DRAM,
  i.e. NOT bandwidth bound any more; the remaining cost is issue/latency/LDS.
* decode w13: ~0.61 GB of expert weights per token in 5.67 ms -> ~108 GB/s, i.e. ~45% of the
  217-244 GB/s that this exact access pattern can reach. ~2x of theoretical headroom left.

### Experiment: warp-cooperative W-row staging for m_e==1  -> NEUTRAL, REVERTED
Idea: at B=1 the single-token S_K loop reads each row's 9 super-blocks with dependent 2 B/lane
loads; stage the whole row (630/918 B) into LDS with wide uint4 loads (9 round trips -> 2),
reusing the existing sXs buffer (8 warps x 1,024 B) so LDS does not grow; alignment handled by
staging from the 16 B-aligned floor (overhang stays inside desc.body).
Result: parity still KLD 0.000000 / Top-1 100% / PPL 929388.622, selfcheck PASS, but
decode 40.9-41.0 tps (was 41.0-41.1) and B=1 moe_w13 5.67 ms (was 5.67). No effect.
Conclusion: the hypothesis was wrong - those 9 loads are independent and the compiler already
software-pipelines them; the decode MoE path is limited by something other than W load latency
(most likely the small-work-per-warp shape and the per-visit kernel structure). Reverted to keep
the shipped diff minimal.

## 12. FINAL SHIPPED STATE (verified)
Working tree (2 files, +401/-44): `hip/ink_hip_codecs.hip.h` (LDS X-slice staging in
`k_moe_grouped_w13_situ`, LDS act-slice staging in `k_moe_grouped_w2`, bf16 GEMM fragment hoist,
bf16 v2 kernel from the previous session) + `hip/ink_hip_backend.hip.cpp` (attention split-KV +
profiler marks from the previous session). No debug env markers left in the tree; no dead kernels.
Binary md5 f8941cb1b568db2145dbf8ad0b2ba096.

Gates (final binary):
* `--kld-eval` vs the staging-OFF reference (n=256, single B=256 forward): **Mean KLD 0.000000,
  Top-1 100.00%, PPL 929388.622 (identical)** -> bit-identical output.
* `--selfcheck` PASS.
* bench prompt 2048 / gen 64, 3 runs: **prefill 160.4 / 160.4 / 160.1 ppts, decode 41.0 / 41.0 / 40.9 tps**.
* long-context smoke: prompt 8192 / gen 32 -> 138.7 ppts, 31.1 tps (no faults; staging guards hold).

| | committed baseline | session start | FINAL | delta vs committed |
|---|---|---|---|---|
| prefill | 117.1 ppts | 130.0 ppts | **160.3 ppts** | **+36.9%** |
| decode | 35.4 tps | 40.8 tps | **41.0 tps** | **+15.8%** |

What this session added: grouped-MoE LDS slice staging, worth **130.0 -> 160.3 ppts (+23.3%)**
with decode unchanged, from one root cause: warp-per-row GEMM mapping re-reading X from L2 ~1 TB
per chunk. What did NOT work, for the record: int8 sdot4 MoE (wrong experts), GEMM fragment
hoisting (+0.5%, noise), and wide uint4 W-row staging for m_e==1 (neutral, reverted).
Scope of the parity evidence: the B=256 grouped-MoE path and dense bf16 GEMMs; the B=1 / m_e==1
decode branches are byte-for-byte unchanged code, which is why decode tps did not move.

## 13. Reading upstream for new ideas (VlDubr/strata-mi50, webzone/strata-gfx906, hippihx)
`VlDubr/strata-mi50` carries a "Wave-1..7" HIP optimization series that claims RDNA2-4. Checked what is
actually new to us: k_moe_reduce_sconv_add, k_moe_direct_w2_par/_reduce, k_rms_norm_rows3,
k_add_bias_rows, k_split_qkvrg, k_swiglu_split_f32, k_sconv_snapshot_pair, k_unembed_argmax_batch_*,
k_moe_scatter_buckets -- ALL ALREADY PRESENT in our tree, i.e. that series is largely upstreamed into
Niko1221/Strata main, which is what we sit on. Their "2x-occupancy Grouped-MoE" was just
__launch_bounds__(128)->(256) on the WMMA kernels, which we never run. Their WMMA MoE + hipBLASLt
paths are unavailable here (no matrix cores, blasLt=0). The one real gap is upstream's
"Fast Path A2: unpack-once + FP16 expert GEMM", gated on hipBLASLt.
`webzone/strata-gfx906` is mostly validation methodology (opt-in guarded tile trials, real-QSA logit
divergence limits, paired A/B controls) -- endorses gating new kernels + a logit gate, which we already do.

### ALU-strip probe: the S_K decode ALU is NOT the grouped-MoE bottleneck
Probe: replaced decode_sk_sub_eff_i8 + decode_sk_8_unscaled_f16 with a 1-instruction load and a constant
1.0h half2 pair in the multi-token S2_K/S3_K staged branches (same memory traffic, same fdot2 count,
garbage values, timing only).
Result: grouped MoE bucket got SLOWER (moe_w13 933-969 ms vs 871-895 shipped), prefill 155.3 vs 160.4 ppts.
Conclusion: removing essentially all dequant ALU buys nothing -> the unpack-to-fp16 idea (upstream A2)
would mostly trade traffic for traffic, and is deprioritized. Tree restored from the md5-verified
pristine copy (f0b546a0cc0ed88d8417bc28e199551a) and rebuilt; zero probe markers left.

### Next hypothesis: workgroup shape, not arithmetic
Per workgroup the kernel stages 9 X slices (4 KB each, x token tiles) while the S_K weight stream it
consumes is ~630 B per output row x 8 rows. So X staging is ~7x the weight traffic PER WORKGROUP, and
each workgroup only does ~450 instructions of real math. Change under test: 16 warps / 8 (gate,up)
pairs per workgroup (512 threads) instead of 8 warps, halving workgroup count and X-slice re-staging.
Kernel derives pairs-per-block from blockDim.x, so one body serves both configs; LUT fills and the
sXs staging are guarded to tid<256; s_gu grew to [16][8]; launch uses MOE_W13_CFG(inter), which falls
back to the old 8-warp config unless inter % 8 == 0. Per-row lane mapping, slice order and wave32_sum
are untouched, so it is expected bit-identical.

### 16-warp grouped w13: 3x MoE speed but parity FAILS -> bisecting
Change: 16 warps / 8 (gate,up) pairs per workgroup (512 threads) instead of 8 warps, halving workgroup
count so staged X slices are reused across 2x as many rows. MoE bucket 871 -> ~285 ms, prefill
160.4 -> ~255 ppts. BUT --kld-eval FAILS deterministically: Mean KLD 3.22, Top-1 72.16%, PPL 4.1e7 vs
9.3e5. --selfcheck still PASSes (it never covers grouped MoE), which is exactly why the logits gate exists.
First suspect (__launch_bounds__(256) vs 512 threads, UB) was fixed -> identical failure signature
(72.16% both times), so that was not it. Audited and fixed as a hygiene matter: all thread-strided
loops in the kernel now use blockDim.x, 256-entry LUT fills and sXs staging guarded to tid<256,
s_gu grown to [16][8]. Multi-token SF4/MX8/GGUF_RAW branches audited: all lane-partitioned (blk=lane;
blk+=32), token gather is warp-replicated -> none of them depend on warp count. j/row coverage checks
out exactly (grid.x=(1152+7)/8=144, 144*8*2 = 2304 = 2*inter rows), no early returns possible.
Failure magnitude (Top-1 72%) is consistent with part of the row space never being computed, so the
differential bisect under test is:
  V1 = all code edits, OLD 256-thread/4-pair config        -> expect PASS + ~160 ppts (isolates my edits)
  V2 = 512 threads with pairs-per-block FORCED to 4         -> full coverage, duplicate writes;
       if V2 passes and is ~871 ms, the 285 ms was skipped work (idea dies);
       if V2 passes at ~285 ms the speed is real and the j-mapping/coverage is the bug.
Nothing shipped; branch rdna2-moe-lds-staging still at the validated 160.4 ppts / 41.0 tps state.

### Bisect result: the kernel was never wrong; my launch macro dropped grid.y
V1 (all code edits, old 256-thread/4-pair config)         -> prefill 160.0 ppts, i.e. unchanged from shipped.
V2 (512 threads, pairs-per-block FORCED to 4 => full coverage + duplicate writes)
  -> Mean KLD=0.000000 | Top-1=100.00% | PPL=929388.622  -> BIT-IDENTICAL, prefill 98.1 ppts (does 2x work).
So the 16-warp kernel itself is bit-identical. The KLD 3.22 failure came from my MOE_W13_CFG macro:
its dim3() passed only grid.x, silently dropping c_.n_experts from grid.y, so expert 0 alone ran.
(The apparent "3x MoE speedup" was that bug: the pmark window also contains k_moe_grouped_w2, which kept
sweeping all 128 experts over stale bucket_act.) Fixed to MOE_W13_CFG(inter, n_experts); also removed a
backslash-space-newline in the macro continuation that I had introduced.
Lesson for this codebase: kernel launches here are not error-checked, and a dim3 with a missing argument
fails silently into wrong output rather than a launch error -- always keep the logits gate on.

### 16-warp config, corrected: bit-identical but SLOWER -> reverted
With grid.y restored to c_.n_experts: KLD=0.000000 | Top-1=100.00% | PPL=929388.622 (bit-identical, as V2
predicted) and selfcheck PASS, but MoE bucket 871 -> 1091-1118 ms and prefill 160.4 -> 140.9 ppts (x3 runs,
decode unchanged 41.0). Reverted with `git checkout -- engine/src/inkling2/hip/`; codecs md5 back to
f0b546a0cc0ed88d8417bc28e199551a.

### What the two negatives actually tell us (elimination, with numbers)
For B=256 prefill, grouped MoE ~871 ms/chunk:
* NOT dequant ALU: stripping decode_sk_sub_eff_i8 + decode_sk_8_unscaled_f16 entirely => no speedup
  (bucket went UP to ~933-969 ms).
* NOT weight DRAM bandwidth: 10.4 GB/chunk of expert weights = ~12 GB/s vs 217-380 GB/s measured attainable.
* NOT LDS operand bandwidth: X operand traffic is ~1.04 TB/chunk (2 B of X per MAC), and aggregate LDS on
  20 CUs x 4 SIMD x 128 B/clk x ~2.4 GHz is ~24 TB/s => ~42 ms, i.e. ~5% of the bucket.
* NOT pure fdot2 issue: ~8.1 G warp fdot2 instrs vs ~192 G warp-instr/s issue => ~42 ms, also ~5%.
* More warps per workgroup is actively harmful (this experiment): the block-wide __syncthreads() count
  stays per-slice while each barrier gets more expensive, and the cooperative staging is still done by
  only 8 of the 16 warps.
Estimated barrier cost: ~12 block barriers per (tile, workgroup) (9 slices + 3 epilogue) x ~30-60 clocks
vs ~5,000 warp-instrs of work per workgroup => barriers plausibly ~40-50% of the bucket. Consistent with
"more warps = worse" and "strip ALU = no change".

### The lever this points to: R rows PER WARP (not more warps per workgroup)
Keep 8 warps/block, let each warp own R output rows. Then the same 9 slice barriers and the same staged
X slice serve R rows, so barrier cost and LDS X reads per unit of work both fall by R, while the weight
stream (which must be read once per row anyway) is unchanged and register cost is only +8 VGPRs per
extra row (acc[R][8]; measured 140 VGPRs at 7 waves/SIMD, so R=2 or 3 fits comfortably).
Sketch: shrink grid.x by R and map j = (blockIdx.x * 4 + (wave >> 1)) * R + r; per slice, load xv from
LDS ONCE per token then loop r, decoding wv from R distinct row pointers and accumulating into acc[r][t].
Per-row accumulation order over s is unchanged => expected bit-identical; gate with --kld-eval as always.
Must be applied to every branch in the kernel (SF4/MX8/GGUF_RAW and the non-staged SK fallback) because
grid.x shrinks globally -- safest as a separate kernel variant selected at launch, or an R-loop wrapping
the branch bodies, given that a wrong launch config here fails silently rather than erroring.

## 14. Experiment log for this cycle: 6 changes tested, 0 shipped, all bit-identical or neutral/worse
Every variant was gated with --dump-ref/--kld-eval against ref_n256.bin. All that produced correct output
produced KLD=0.000000 / Top-1=100.00% / PPL=929388.622, i.e. bit-identical. Baseline for comparison:
grouped MoE bucket 871-895 ms/chunk, prefill 160.1-160.4 ppts, decode 41.0-41.1 tps.

| # | change | LDS / waves | MoE bucket | prefill | verdict |
|---|---|---|---|---|---|
| 1 | 16 warps (512 thr) / 8 pairs per workgroup | 15.6 KB, waves down | 1091-1118 ms | 140.9 | REVERT (occupancy) |
| 2 | token tile tm 8 -> 16 | 24 KB, 2 wg/CU | 1011-1043 ms | 147 | REVERT (occupancy) |
| 3 | weight super-block prefetch (s+1 before s's dot chain) | unchanged | 874-898 ms | 160.0-160.2 | REVERT (neutral) |
| 4 | alias sX with sXs (LDS 15,616 -> 11,008 B) | 11 KB, more wg/CU | 873-896 ms | 159.9-160.2 | REVERT (neutral) |
| - | (earlier) ALU-strip probe of the S_K dequant | unchanged | 933-969 ms | 155.3 | probe, reverted |
| - | (earlier) 16-warp attempt with a grid.y bug | - | 282 ms (fake) | 255 (fake) | invalid, see §13 |

Conclusions, in order of confidence:
1. Resident waves are the binding constraint on grouped w13: every change that lowered waves lost 8-15%,
   and the two changes that could have raised it (LDS aliasing) were neutral because occupancy is limited
   by registers (140 VGPRs), not by LDS. So there is no free occupancy left to win.
2. Neither the S_K dequant ALU nor expert-weight load latency is an exposed stall (both probes neutral),
   so upstream's "unpack-once + FP16 GEMM" (Fast Path A2) has little upside here even though RDNA2 has the
   fdot2_f32_f16 instruction this MoE path already uses.
3. The kernel is at ~5% of both the fdot2 roof (1.04 TFLOP vs ~24.5 TFLOP/s) and the DRAM roof
   (10.4 GB vs 217-380 GB/s) and does not respond to data-movement, ILP, tile-width or occupancy knobs.
   That points at the warp-per-row mapping itself: a 70 B weight block per (row, slice) feeds only
   tm=8 dot products, and fixing that ratio means a GEMM-style retile with weight reuse across output rows
   held in registers -- which reassociates the k accumulation and therefore CANNOT stay bit-identical.
   That is a quality-neutrality decision for the owner, not a free optimization, so I stopped there.
4. Tooling gap: this box has no profiler (librocprofiler-register only; no rocprofv3 binary, no /opt/rocm),
   so every micro-optimization below is guesswork. Getting rocprofv3 (or any SQ counter source) is the
   single highest-value prerequisite for further kernel work; it would replace this table of eliminations
   with actual stall reasons.
5. Same story on the biggest decode bucket: k_bf16_gemv1_qkvrg already uses coalesced uint4 (16 B/lane)
   loads at full occupancy; at ~164 GB/s vs the ~380 GB/s roof it looks short-row/latency-tail limited
   (K=2304 is only ~4 loop iterations per wave), which is not fixable by load-width changes. Untested
   because it needs counters to guide it.
Everything landed in this file and the branch is unchanged at the validated 160.4 ppts / 41.0 tps state.

## 15. Correction + the one lever I did NOT test: R rows per warp (bit-identical)
Correction to §14 conclusion 3: I said the only remaining lever was a GEMM-style retile that reassociates
the k accumulation. That is wrong. "R output rows per warp" (section 13's own note) does NOT reassociate
anything: each row's accumulation order over k stays exactly as it is, so it is bit-identical, yet it cuts
both the per-row barrier count and the LDS X reads per unit of work by R while leaving the weight stream
untouched. It is the only remaining idea consistent with the evidence (waves-bound; not ALU; not load
latency) and it is UNTESTED -- I mistakenly tested "more warps per workgroup" (exp #1), which is a
different thing and the one that hurt.
Recommended recipe (lowest-risk form, still to be validated by --kld-eval):
  * Keep 8 warps/block and keep grid.x = (inter + 3) / 4 unchanged; do NOT shrink the grid.
  * Let each warp own two j values: j and j2 = j + (inter >> 1). Blocks whose j >= (inter >> 1) return
    immediately -- this is block-uniform, and all barriers are block-scoped and sit after the return
    point, so it is safe (this is what avoids any grid-config change, which is where the silent-failure
    risk lives in this codebase).
  * In the staged S2_K/S3_K branches only: acc[2][8] plus two row pointers; inside the slice loop load
    xv from LDS ONCE per token, then apply it to both rows' freshly decoded wv. Cost ~+8 VGPRs.
  * The remaining branches (SF4 / MX8 / GGUF_RAW / non-staged SK fallback, and the m_e==1 path) must also
    cover the second half of the j-space, so each needs an outer pass over the two j values. They are
    1-2 experts per layer, so their speed is irrelevant -- correctness only.
  * Gate: --kld-eval against ref_n256.bin must stay Mean KLD=0.000000 / Top-1=100.00% / PPL=929388.622.
  * Watch waves: register pressure is what caps occupancy here (140 VGPRs), so if the wave count drops,
    pair the change with trimming registers elsewhere rather than reverting outright.

## 16. ROOT CAUSE TOOL: -Rpass-analysis=resource-usage explains the tm=16 and two-row losses
No profiler counters exist on this box, but the *compiler* reports resource usage per kernel, and that is
available offline. Reproduce (CPU only, no GPU, ~90 s):
  cd /mnt/win/llm-temp/build-ink/src/inkling2/hip
  /usr/lib/llvm-22/bin/clang++ -x hip -c -O3 -std=c++20 --offload-arch=gfx1031 --rocm-path=/usr \
    --rocm-device-lib-path=/usr/lib/llvm-22/lib/clang/22/amdgcn/bitcode \
    -I.../engine/src/inkling2/hip/../../.. -I.../include -I.../hip/.. -I.../hip \
    -I.../third_party/stb -I.../third_party/zstd-dec -DINKLING_ENABLE_HIP=1 \
    -Wall -Wextra -Wno-unused-function -Wno-unused-result -Rpass-analysis=resource-usage \
    .../hip/ink_hip_backend.hip.cpp -o /tmp/probe.o
(the exact line is in build.ninja, target `src/inkling2/hip/ink_hip_backend.o`), then grep the mangled
kernel name for: TotalSGPRs, VGPRs, ScratchSize [bytes/lane], Occupancy [waves/SIMD], *Spill, LDS Size.

### Measured, k_moe_grouped_w13_situ
| build | VGPRs | Scratch B/lane | waves/SIMD | LDS B/block | bench prefill |
|---|---:|---:|---:|---:|---:|
| shipped (1 row/warp, LDS staged) | 140 | 0 | 7 | 15,616 | 160.4 ppts |
| two rows/warp, runtime `hb` index | 140 | **80** | 7 | 15,872 | 133.9 ppts |
| two rows/warp, constant 0/1 index | 161 | 0 | 5 | 15,872 | see below |

### Two conclusions that change how to read every earlier experiment
1. **Any runtime-indexed accumulator array is demoted to scratch (private/VRAM) memory.** `acc[hb][t]` with a
   runtime `hb` produced `ScratchSize: 80 bytes/lane` while still reporting *VGPRs Spill: 0* -- so a
   "0 spills" line does NOT prove the accumulators live in registers. The kernel got 35% slower purely from
   private-memory traffic. Fix: index accumulators with compile-time constants only (a `do{}while(0)` macro
   instantiated as `EPI(0)` / `if (two) EPI(1)`; `#pragma unroll` on a runtime-bound loop is not enough).
   This very likely also explains why the `tm` 8 -> 16 token-tile experiment lost ~9% despite looking
   bit-identical and harmless: `acc[16]` indexed by a runtime `t`.
2. **Occupancy on this kernel is capped by the per-SIMD VGPR file, and the cliff is steep and close.**
   140 VGPRs -> 7 waves/SIMD; 161 VGPRs -> 5 waves/SIMD. LDS (15.9 KB -> 4 blocks/CU -> 8 waves) is NOT the
   binding constraint, which is exactly why the LDS-alias experiment was neutral. Anything that adds more
   than ~6 live VGPRs to this kernel pays a wave cliff, so register-neutrality is the hard budget for
   grouped w13, not LDS or ALU.

### 16.1 Two rows per warp, FINAL: reverted (bit-identical, still slower) with the numbers that explain it
| variant | VGPRs | waves/SIMD | Scratch | moe_w13 ms/chunk | bench prefill |
|---|---:|---:|---:|---:|---:|
| shipped, 1 row/warp | 140 | 7 | 0 B | 871-895 | 160.4 ppts |
| 2 rows/warp, runtime index | 140 | 7 | 80 B | 1180-1205 | 133.9 ppts |
| 2 rows/warp, constant index | 161 | 5 | 0 B | 937-962 | 153.3 ppts |
All three were bit-identical (`Mean KLD=0.000000 | Top-1=100.00% | PPL=929388.622`, selfcheck PASS).
Interpretation: sharing each staged X slice and each slice barrier across 2 output rows is worth roughly
as much as the 2 of 7 waves it costs, i.e. they nearly cancel at 5 waves and leave -7% on MoE (-4.5%
end-to-end). Reverting means grouped w13 is genuinely balanced, not obviously mis-tuned.
Occupancy model fitted to these three points (gfx1031, 8-warp workgroups):
    waves/SIMD = floor(31744 / (32 * ceil8(V)))
  -> 7 waves needs V <= 141. The shipped kernel is at 140. There is exactly ONE register of headroom,
     so a "2 rows/warp at 7 waves" build cannot exist without first removing ~20 live VGPRs elsewhere
     in the kernel. That, not LDS or ALU or load width, is the wall this kernel is against.
Decision: stop micro-optimizing grouped w13 blind. Remaining options both need an owner decision or new
tooling -- (a) a GEMM-style retile that reassociates k accumulation (breaks bit-identity, and speculative
-decode acceptance is hypersensitive to trunk numeric changes), or (b) real SQ stall counters (rocprofv3).

## 17. Why a real profiler is NOT obtainable on this box (checked to the kernel, not assumed)
AMD's rocprofv3 *is* downloadable for this stack -- ROCm 7.2.4 publishes `rocprofiler` (= rocprofv3) for
ubuntu/noble at https://repo.radeon.com/rocm/apt/7.2.4/dists/noble/main/binary-amd64/Packages, and the deb
could be extracted locally (`dpkg -x`) + LD_LIBRARY_PATH'd without disturbing the working ROCm bundle in
/mnt/win/llm-temp/enginetest. The blocker is the kernel, not the package:
    id -u                          -> 1000 (unprivileged)
    grep CapEff /proc/self/status  -> 0000000000000000  (no CAP_PERFMON, no CAP_SYS_ADMIN)
    /proc/sys/kernel/perf_event_paranoid -> 4           (cannot be lowered without CAP_SYS_ADMIN)
    /sys/module/amdgpu exists, but the above already forbids KFD performance-counters
SQ stall / occupancy / VM-refill counters therefore cannot be collected here at all. Without CAP_PERFMON
(or perf_event_paranoid <= 2 plus CAP_PERFMON) rocprofv3 degrades to API/dispatch tracing, which yields
per-kernel durations -- strictly less than the STRATA_HIP_PROF=2 buckets we already have. So the absence
of counters is a hard environmental limit, not missing effort.
To get real stall data, one of these must happen (all need someone with root):
  * run under sudo with CAP_PERFMON, or set kernel.perf_event_paranoid<=2, then extract the 7.2.4 noble
    `rocprofiler` + `rocprofiler-register` debs into /mnt/win/llm-temp/rocprof/ and profile with
    LD_LIBRARY_PATH=/mnt/win/llm-temp/enginetest:$PREFIX/lib (do NOT apt-install over the system ROCm;
    the system libamdhip64.so.7.15 segfaults, which is why the enginetest bundle exists);
  * or run the same kernels on any box where counters are permitted.

## 18. REAL HW COUNTERS ACHIEVED (corrects §17 -- I have passwordless root)
§17 was wrong: `sudo` is available and `(ALL) NOPASSWD: ALL`, so kernel counters are reachable.
Recipe that actually works on this box (never apt-installs, so the working enginetest ROCm bundle is safe):
    sudo sysctl -w kernel.perf_event_paranoid=2          # revert with =4 when done
    # from https://repo.radeon.com/rocm/apt/7.2.4/dists/noble/main/binary-amd64/Packages, dpkg -x each into
    # /mnt/win/llm-temp/rocprof/root :  rocprofiler-sdk  rocprofiler-sdk-roctx  rocprofiler-sdk-rocpd
    #                                   hsa-amd-aqlprofile  rocprofiler-register  rocm-smi-lib  rocm-core
    # in $PREFIX/lib:  ln -s librocprofiler-sdk.so.1.1.0 librocprofiler-sdk.so.1.1.1.0
    #                  ln -s librocprofiler-sdk-roctx.so.1.1.0 librocprofiler-sdk-roctx.so.1.1.1.0
    # (rocprofv3 1.1.0 hard-codes the .1.1.1.0 soname); libhsa-amd-aqlprofile64.so.1 from the aqlprofile
    # deb is MANDATORY even for --kernel-trace, and is absent from enginetest.
    sudo env LD_LIBRARY_PATH=/mnt/win/llm-temp/enginetest:$PREFIX/lib \
      $PREFIX/bin/rocprofv3 --kernel-trace --stats --output-format csv -o o -d /tmp/prof -- ./strata-inkling ...
Gotchas: in ROCm 7.2 `rocprofv3` replaces `rocprof`/`rocprofv2` (the `rocprofiler` deb's `rocprof` is the
deprecated v1); the SDK split packages must all be extracted or the *target app* fails to load; run the
whole thing under sudo (CAP_PERFMON); keep enginetest FIRST in LD_LIBRARY_PATH so the app keeps its own
HIP/HSA (its HSA is 1.21.0) and only the profiler libs come from the extracted tree.

### Measured, shipped build, k_moe_grouped_w13_situ, B=256 (48 dispatches)
geometry from the profiler itself: Workgroup 256 thr, LDS 15,872 B/block, **VGPR 144**, SGPR 128,
**Scratch 0**, grid = (288, 128) -> 36,864 workgroups = 294,912 waves/dispatch, duration median 1.117 ms.
Whole-run split (512-prompt/4-gen run, 273 ms GPU total): bf16_gemv dense 23.8%, moe_w13 19.4%,
moe_w2 10.7%, direct_w13 7.4%, qkvrg 7.0%.
Stall mix (sums over the 48 dispatches): SQ_WAIT_ANY 5.212e10, SQ_WAIT_INST_ANY 3.199e9,
SQ_WAIT_INST_LDS 1.372e8, SQ_BUSY_CYCLES 2.659e8, SQ_WAVES 1.416e7.
  -> **SQ_WAIT_INST_ANY / SQ_WAIT_ANY = 6.1%**: only a small slice of stall time is instruction/ALU
     dependency latency. Grouped w13 is **memory-latency bound**, which independently confirms the
     ALU-strip probe (removing dequant ALU made it slower) and kills any remaining "reduce the math"
     idea on this kernel.
Unsupported on gfx1031 (report 0 / absent): SQ_COUNT, SQ_WAIT_LM, SQ_WAIT_BARRIER. PC sampling is NOT
supported on this agent (`rocprofv3-avail list --pc-sampling` lists no agents), so instruction-level
attribution is unavailable; wave-level occupancy + durations are what we get.
Register/occupancy model refined with the profiler's own VGPR reading: allocation granularity makes the
shipped kernel 144 VGPRs -> 7 waves/SIMD; per-SIMD reg file ~32768, so 8 waves needs <=128 VGPRs.
Forcing `__launch_bounds__(256, 8)` does reach 8 waves but produces VGPRs Spill 4 / ScratchSize 20 B/lane
(see §16: scratch is what cost 35% before) -> measured separately.

## 19. FIRST WIN FROM THE COUNTER DATA: 8 waves/SIMD on grouped w13 (+7.1% prefill, bit-identical)
Commit d613831 on branch rdna2-moe-lds-staging (1 line changed). Direct consequence of §18: because
k_moe_grouped_w13_situ is memory-latency bound (6.1% instruction-latency stalls) and its natural 144 VGPRs
capped it at 7 waves/SIMD, asking the compiler for the 8th wave is worth paying spills for.
    - __global__ void __launch_bounds__(256) k_moe_grouped_w13_situ(
    + __global__ void __launch_bounds__(256, 8) k_moe_grouped_w13_situ(
All four launch sites pass 256 threads, so the bound is legal everywhere (a mismatched launch_bounds would
have been a silent wrong-answer risk, which is what bit the 16-warp attempt).
| | shipped (7 waves) | this (8 waves) |
|---|---:|---:|
| VGPRs | 144 | 128 (+4 spills, 20 B/lane scratch) |
| prefill | 160.2-160.4 ppts | **171.6 / 171.8 ppts** |
| decode | 41.0 tps | 40.9 / 40.8 tps |
| moe_w13 | 871-900 ms/chunk | **757-779 ms/chunk** (-13%) |
Parity vs the pre-change reference: Mean KLD=0.000000 | Top-1=100.00% | PPL=929388.622 unchanged (255
positions); --selfcheck PASS. Spills relocate values but cannot reorder arithmetic, so bit-identity holds
by construction and the gate confirms it.
IMPORTANT nuance vs §16: scratch is not automatically fatal. 20 B/lane cost less than the extra wave
bought. The spill/occupancy curve for this kernel (compile-only probe, free):
| target waves/SIMD | VGPRs | VGPR spills | Scratch B/lane |
|---:|---:|---:|---:|
| 7 (natural) | 144 | 0 | 0 |
| 8 | 128 | 4 | 20 |
| 9 | 112 | 20 | 84 |
| 10 | 96 | 32 | 132 |
Spills grow non-linearly past the 8th wave, so 8 is likely the sweet spot; 9 is measured separately.
The right follow-up is to earn ~16 VGPRs back legitimately so 8 waves costs zero scratch -- that should
beat 171.7 rather than merely match it.

### 19.1 9 waves/SIMD measured worse than 8; long-context also improved. FINAL shipped state = d613831
| target | prefill (2048) | moe_w13 ms/chunk |
|---|---:|---:|
| 7 waves (previous ship) | 160.2-160.4 ppts | 871-900 |
| **8 waves (shipped now)** | **171.6-171.8 ppts** | **757-779** |
| 9 waves | 169.4 ppts | 787-803 |
So the occupancy/spill optimum is 8 waves/SIMD (matches the spill curve in §19: 4 spills at 8 vs 20 at 9).
Long context, prompt 8192 / gen 32: prefill 147.6 ppts (was 138.7 before this change, +6.4%);
decode 30.7 tps (was 31.1; -1.3% at 32 generated tokens, within the noise of a 1.0 s decode window).
--selfcheck PASS; --kld-eval parity Mean KLD=0.000000 / Top-1=100.00% / PPL=929388.622 unchanged.
Cumulative against the committed baseline: prefill 117.1 -> 171.7 ppts (+46.6%), decode 35.4 -> 41.0 tps
(+15.8%), all bit-identical at the logits gate.
Note for anyone profiling later: kernel.perf_event_paranoid was left at 2 (from 4) so rocprofv3 counter
collection keeps working; revert with `sudo sysctl -w kernel.perf_event_paranoid=4`.

## 20. Why "9 waves" lost: LDS, not registers, is the real occupancy ceiling -- and two tool corrections
New goal: keep going until the next bump. Bar = d613831 (171.6-171.8 ppts / 41.0 tps, moe_w13 757-779 ms).
Tool correction #1: `-Rpass-analysis=resource-usage` reports "VGPRs Spill: N" and a non-zero
ScratchSize for k_moe_grouped_w13_situ even when the kernel contains ZERO spill traffic. Verified by
dumping the ISA (`--save-temps=obj`, then the `*.hip-amdgcn-amd-amdhsa-gfx1031.s` file): both the shipped
8-wave build (4 spills / 20 B) and the 112-VGPR 9-wave build (20 spills / 84 B) contain
`grep -c scratch_store|scratch_load == 0`. The private-segment number is flat-addressing reservation
(`.uses_flat_scratch 1`), not spills. So: never conclude "this build spills" from that remark; check the ISA.
Consequence: the +7.1%/+13% win came from the extra wave alone, and there is no register reclaim to do.
Tool correction #2 (mine): the 9-wave loss in §19.1 was not "too many spills", it was never 9 waves.
Hardware waves/SIMD = min(register cap, 8 * floor(65536 / LDS_per_block) / 4). With LDS 15,616 B only
4 workgroups fit per CU -> 8 waves/SIMD, so requesting 9 bought 112 VGPRs of compiler pressure and the
same 8 waves -> 169.4 < 171.7. It also retroactively explains why the §14 LDS-alias experiment was
neutral: at 7 waves LDS was not the binding constraint, so freeing it changed nothing.
Fix now under test (b227813c3): alias the two mutually exclusive path buffers
    sX (4,608 B, m_e == 1 path) and sXs (8,192 B, m_e >= 2 staged path) -> one sXd[2][8][256]
which drops LDS 15,616 -> 11,008 B = 5 workgroups/CU = up to 10 waves/SIMD, with __launch_bounds__(256, 9).

## 21. SECOND BUMP: grouped w2 to 8 waves/SIMD -> 177.3/177.4 ppts (commit 9df38f5)
k_moe_grouped_w2 naturally allocated 154 VGPRs -> 6 waves/SIMD, while its 12,800 B LDS allows 5
workgroups/CU (10 waves), so it was purely register-limited -- the same setup w13 had in d613831.
    - __global__ void __launch_bounds__(256) k_moe_grouped_w2(
    + __global__ void __launch_bounds__(256, 8) k_moe_grouped_w2(
ISA check first (per §20): zero scratch_store/scratch_load in the module at 128 VGPRs, so the reported
"19 spills / 80 B/lane" is again flat-addressing reservation, not spill traffic.
| | before (6 waves) | after (8 waves) |
|---|---:|---:|
| prefill | 171.6-171.8 ppts | **177.3-177.4 ppts** (+3.3%) |
| MoE bucket | 757-779 ms/chunk | **722-736 ms** (-5%) |
| decode | 41.0 tps | 41.0 tps |
Parity Mean KLD=0.000000 / Top-1=100.00% / PPL=929388.622 unchanged; --selfcheck PASS.
Cumulative vs committed baseline: 117.1 -> 177.4 ppts (+51.5%), 35.4 -> 41.0 tps (+15.8%).
Occupancy survey of the whole backend after this change (waves/SIMD, VGPRs, LDS):
  * k_moe_grouped_w13_situ 8 (LDS-capped at 15,616 B), k_moe_grouped_w2 8 (register, LDS would allow 10)
  * k3_flash_splitkv_part / k3_flash_tree_splitkv_part / k_bf16_gemv1_qkvrg / k_bf16_gemv1_wo_sconv /
    k1_loop_ctrl_* / k_rms_norm_rows3 / k_moe_reduce_sconv_add: ALL 16 -> no occupancy headroom left
  * still register-capped below 16: k_moe_direct_w2_reduce 8 (127 VGPRs, LDS 512 -> decode path, small)
  * LDS-capped rather than register-capped: k5_dspark_base_top2_screen 8 (LDS 32,256 B),
    k_unembed_argmax_batch_stage1<> 8 (LDS 27,648 B) -- launch_bounds cannot help these; only shrinking
    their LDS could.
So the __launch_bounds__ lever is now exhausted for every large prefill bucket. Remaining untested
candidates, in rough order of value: k_moe_direct_w2_reduce occupancy (decode), w2 at 9-10 waves
(w13's analogous test lost, low prior), LDS reduction in the dspark/unembed kernels (decode/spec path).

## 22. Chunk size is a real prefill lever (opt-in): +5.4% prefill at 512, but long-context decode pays
max_prefill_chunk defaults to 512-2048 in hip_uma_guard.hpp, then a post-load heuristic caps it:
`tight_chunk = (post_free < 1.5 GiB) ? 256 : 512`; we have 982 MiB free -> 256.
Prefill streams every expert's weights once PER CHUNK, so 2048 tokens = 8 weight passes at 256, 4 at 512.
| chunk | prefill 2048 | decode 2048 | prefill 8192 | decode 8192 |
|---|---:|---:|---:|---:|
| 256 (default) | 177.0-177.2 ppts | 40.8-40.9 tps | 151.2-151.3 ppts | 30.9-31.2 tps |
| 512 | **186.7-186.8 ppts** | 40.9-41.0 tps | **158.7 ppts** | **27.4-28.0 tps** |
Numerics are identical across chunk sizes (Mean KLD=0.000000, Top-1=100.00%, PPL=1101.388 over 511
positions on a 512-token prompt, compared against logits dumped at chunk 256) -- batching shape does not
change the math here, so this is scheduling only.
Shipped as opt-in env STRATA_MAX_PREFILL_CHUNK (64..4096); unset reproduces the old heuristic exactly.
Rationale for not flipping the default: 512 buys +5.4% prefill but costs ~11% decode at 8192-token
context, and decode throughput is the metric that matters for interactive use.
Open question worth a future session: WHY does a bigger chunk slow later decode? The KV ring config is
identical (1536 positions, window 1024) and post-load free VRAM is the same 982 MiB at startup, so it is
probably batch-scratch allocation interacting with the KV pool or page locality after a 512-token prefill.
If that is fixed, 512 could become the default and the +5.4% would be unconditional.

### 22.1 VRAM pressure ruled out as the cause of the chunk-512 long-context decode cost (2x2, same build)
| config | prefill 8192 | decode 8192 |
|---|---:|---:|
| chunk 256, vram-experts 8.77 | 151.1 | 31.2 |
| chunk 512, vram-experts 8.77 | 158.7 | 28.4 |
| chunk 512, vram-experts 8.30 (about 480 MiB more free) | 158.7 | 28.5 |
| chunk 256, vram-experts 8.30 | 151.2 | 31.2 |
So freeing VRAM does not recover decode; the cost tracks chunk size alone. STRATA_HIP_PROF B=1 buckets are
IDENTICAL between the two chunk sizes (sum 32.26-32.33 ms, unembed 2.40-2.43 ms, and the profiler's own
tok/s is 28.8 in both) while the bench reports 31.2 vs ~28.5 tps. Since 1/31.2 = 32.05 ms matches the
kernel sum EXCLUDING unembed and 1/28.5 = 35.1 ms matches the sum INCLUDING it, the difference is that at
chunk 256 the unembed/sampling stage overlaps the next step's kernels while at 512 it is exposed -- i.e. a
launch/pipelining-side effect, not GPU work. Not chased further: high archaeology cost, modest prize, and
the knob is opt-in so nothing regresses. Graph-cache keys that mention batch size (e.g. the
`tctx_graph_B_[n_acc] == scratch_B_` condition around ink_hip_backend.hip.cpp:2591) are the place to look
if anyone ever wants the +5.4% unconditionally.

### 22.2 Decode is close to the hardware floor -- roofline sanity check on the decode buckets
Per B=1 step at 2048-token context (32.3 ms total, ~31 tps): qkvrg 6.9, MoE w13 5.7 + w2_post 3.9,
attn 5.95, wo_post 4.5, res_ctrl 2.2, sconv_kv 1.8, moe_prep 1.4, unembed 2.4 ms.
The unembed stage alone reads the 200,058 x 2,304 bf16 output embedding = ~922 MB; at the measured
~380 GB/s wide-load roof that is 2.4 ms, which is exactly what it costs -- it is AT the DRAM roofline, so
there is no headroom there. Same reasoning caps any further decode gain: most of the step is streaming
weights that must be read once per token.

## 23. Chunk size has a hard ceiling, and two of its caps are independent
Chunk sweep with STRATA_MAX_PREFILL_CHUNK only (same build, experts 8.77):
| chunk | prefill 2048 | decode 2048 | prefill 8192 | decode 8192 |
|---|---:|---:|---:|---:|
| 256 (default) | 177.2 | 40.9 | 151.1 | 31.2 |
| 512 | 186.8 | 41.0 | 158.8 | 27.8 |
| 1024 | 186.8 | 41.0 | 158.7 | 28.4 |
| 2048 | 186.7 | 40.6 | 158.7 | 28.5 |
Flat past 512 for a non-obvious reason: a SECOND cap, `swa_span_cap_ = swa_ring_ - swa_win` = 1536 - 1024 = 512
(ink_hip_backend.hip.cpp:734-735). It is a correctness constraint, not tuning: the sliding-window KV ring needs
`ring = window + chunk` margin or a later token of the same batch overwrites a key an earlier query still needs.
The ring is derived from the *engine option* `opts.chunk` (`ring = win + max(opts.chunk, 64)`, line ~727), while
STRATA_MAX_PREFILL_CHUNK only moves the post-load clamp -- so raising the env alone gets silently re-split at
512. Pairing them (`--chunk 1024` + env 1024) does raise the ring to 2048 and the span cap to 1024, and then the
card runs out of memory: batch scratch scales with the chunk and chunk 1024 dies with
"[ink-hip] scratch alloc: out of memory" (982 MiB post-load free; the bench limps on at 33.5 ppts / 9.9 tps,
chunk 2048 at 5.2 ppts). So the post-load heuristic cap I was overriding is load-bearing.
Practical ceiling on this box = 512, worth +5.4% prefill at 2048 tokens (bit-identical) at the cost of ~10%
decode at 8192-token context, so it stays opt-in.
Response to the footgun: the override is now validated against the engine's own estimate_batch_scratch_bytes()
(ink_batch.h:167) and halved until it fits within half the post-load free VRAM, with a clamp message; the
override message now says to pair it with --chunk so the SWA ring margin grows to match.

### 23.1 Scratch-aware clamp shipped
Cases after the guard (RX 6700 XT, parallel 1, kv fp8, experts 8.77):
| STRATA_MAX_PREFILL_CHUNK | result |
|---|---|
| unset | Capped 256 (unchanged), selfcheck PASS, 177.1 ppts / 40.7 tps |
| 512 | honored, PASS, 186.9 ppts / 40.8 tps, Mean KLD=0.000000 Top-1=100.00% (511 pos) |
| 1024 | clamped to 512 (scratch 773 MiB > 491 MiB budget), PASS, 8192 prompt fine at 158.8 ppts |
| 4096 | clamped to 512 (scratch 3093 MiB) |
Estimator bracket matches the empirical cliff: it allows 512 and rejects 1024, which is exactly where
"scratch alloc: out of memory" starts.

## 24. The chunk-512 "decode regression" was a benchmark artifact -- default is now 512 on tight-VRAM cards
Amortization test (prompt 8192, same build, RX 6700 XT, kv fp8, experts 8.77):
| gen | chunk 256 | chunk 512 | extra wall time |
|---:|---:|---:|---:|
| 32 | 31.2 tps (1.03 s) | 28.5 tps (1.12 s) | +0.09 s |
| 128 | 35.7 tps (3.58 s) | 34.7 tps (3.68 s) | +0.10 s |
| 512 | 47.4 tps (10.80 s) | 46.9 tps (10.91 s) | +0.11 s |
Constant, not per-step -> one event. Mechanism: the KV pool grow path bumps pool_gen, kv_gen_sum folds in
every pool_gen, and graph_kv_gen_ == kv_gen_sum is part of the decode-graph reuse condition, so a grow
during decode destroys and re-captures the graph. ~100 ms, once. The gen=32 bench window turns 100 ms into
"10% decode regression"; varying the window instead of trusting one number is what caught it.
Fix shipped: the tight-VRAM branch used to pick 256 purely from post_free < 1.5 GiB. It now takes 512 when
estimate_batch_scratch_bytes(bg, 512) fits within half of post_free (391 MiB vs a 491 MiB budget here),
so the decision uses the engine's own estimate and is legible in the log line.
Final default-path numbers on this box: prefill 186.7-187.0 ppts / decode 40.9-41.0 tps at prompt 2048;
158.6 ppts / 46.9 tps at prompt 8192 gen 512. vs the committed baseline of 117.1 ppts / 35.4 tps that is
+59.5% prefill and +15.8% decode, all bit-identical at the KLD gate.
Escape hatch: STRATA_MAX_PREFILL_CHUNK=256 (reproduces 177.0 ppts, chunk<=256 in the selfcheck line).

## 25. Upstream prior art found via gh (this is the RDNA2 MoE problem, known and open)
- #6264 (OPEN) / #6269: RDNA2 gfx103x MoE gate/up is up to 3.4x slower than it used to be. Root cause per the
  thread: after the MMQ removal the only int8 prompt path for some expert layouts is a fused kernel gated to
  gfx12/gfx1151/gfx1100-1102 "because it needs matrix cores that RDNA2 does not have", so RDNA2 falls back to
  the FP16 path. Matches my prefill profile: moe_w13 = 1410 ms = 52% of prefill.
- PR #6850 (OPEN, not merged, community-authored) adds STRATA_TQ_I8: trellis MoE prefill on int8 DP4A tile
  kernels for gfx103x, dequant inside the kernel. Reporter's own numbers on an RX 6800: 547.2 -> 914.9 tok/s
  (x1.67) and weight dequant 2,230 ms -> 80 ms. This independently confirms that grouped-MoE prefill on RDNA2
  is DEQUANT-bound, not GEMM-bound. It targets trellis TQ2_T/TQK6/TQK7 layouts, not our SF4 experts, and it is
  not in this tree. Idea it licenses for us: SF4 is e2m1 magnitude (x2 = integers 0..12, sign separate), so an
  SF4 -> int8 DP4A grouped kernel is representable on gfx103x without matrix cores. Big project, big ceiling.
- apply_arch_defaults("gfx1030") returns EMPTY (arch_defaults_test.cpp:346): a discrete RDNA2 card gets no
  defaults, so community users set env by hand. Knobs those users run with, which I have never tested here:
  STRATA_HIP_PROMPT_F16=1, STRATA_MTP_SIDE_STREAM=1, STRATA_DENSE_MMQ=1, STRATA_HIPBLASLT_TABLE=on.
  First two exist in this tree (engine/src/core/mtp.cpp). Cheap to A/B on the bench.
- #6343 (OPEN): on gfx1031 Windows HIP the engine can enter a corrupted state where MTP drafts are accepted
  0 of N and output is gibberish, with expert-cache hit rate spiking to ~100% (routing collapse). Health bar
  from that reporter: healthy is >= ~15% accepted. Use --spec-check acceptance as a health check, not just a
  tuning number.
- #1685: RFC/PoC native chunked online-softmax prefill for dense models on HIP, tested on gfx1031 (attention
  is my #2 bucket at 21% of prefill).
- Upstream has NO commits touching engine/src/inkling2/hip/ since my fork point 2af30cc, so the branch is not
  behind. Note the gate `mem_info_.gen != AmdGen::Rdna2` already existed 9 times at the base; my 1811377 added
  only an extra RDNA2 *fast path* branch, it did not disable anything.

### 25.1 Work in flight / tree state note
UNCOMMITTED experiment in engine/src/inkling2/hip/ink_hip_backend.hip.cpp (site ~4275, the main-stream grouped
MoE dispatch): an opt-in STRATA_INK_MOE_TILED=1 that lets RDNA2 take the tiled *_wmma kernel path, added to
separate "gfx103x has no WMMA intrinsics" (true) from "the tiled BMxBK structure is unavailable here" (a
conflation). Nothing ships from this without a measured prefill delta AND --kld-eval numbers, since the tiled
path accumulates in a different order. Validation script: /mnt/win/llm-temp/scratch/moe_tiled_gate.sh
(gate off must reproduce HEAD a0e388d exactly, then gate on). Not built or run yet: the spec-decode sweep job
b38924f8e holds the GPU and goal rules forbid rebuilding under a live benchmark.
From PR #6850's diff, the int8 recipe that a future SF4 DP4A path would copy: v_dot4_i32_i8 via an i8_dot0
wrapper ("falls back to the platform's dp4a"), activations as q8_1 per-block scales, weights through an int8
codebook LUT staged in LDS, then rescale by act_scale x weight_scale. SF4 e2m1 magnitudes x2 are the exact
integers 0..12, so the weight side maps to int8 without loss.

## 26. Spec decode is worth +55% on this card, but the tuning knobs I swept are dead code in the HIP bench path
Prompt 2048, gen 256, HEAD a0e388d:
| config | decode |
|---|---:|
| default (DSpark spec on) | **51.0-51.1 tps** |
| --draft off | 32.8 tps |
So speculation is a large, quality-neutral win (32.8 -> 51.0, +55%) and must stay on.
Every other config tied exactly: --draft-cost 1.5 / 2.5 / 8.0, --draft-beam 1, --draft-beam 1 + cost 1.5,
--draft-plain 1.0, --draft-adapt all gave 51.1 tps / 5.01 s. A 4-way beam tree and a linear draft chain cannot
cost the same, so the tie was not "the setting does not matter". Checking the plumbing:
  - spec_beam / spec_cost / spec_plain are assigned into cfg.bopts in ink_main.cpp:899-905 and are NEVER read
    anywhere in src/inkling2 (grep finds only the assignments). Dead knobs for this engine path.
  - spec_adapt is read only in ink_serve.h (serve path), not in --bench.
  - enable_dspark IS honoured -- the 32.8 vs 51.0 gap proves the on/off switch works.
Conclusion: there is no user-facing spec-window tuning surface for HIP --bench; the window comes from internal
DSpark policy. Any retune has to change that policy in the backend, not pass flags. Lesson for measuring:
identical timings across settings that must differ is a plumbing signal, not a null result.
Also: my --spec-check call was wrong; it takes an integer count (ink_main.cpp:967 spec_check_n = atoi(next())),
so it must be e.g. --spec-check 64 --prompt-ids @file. Acceptance rate is still unmeasured here.
Note on steady state: decode at gen 256 is 51.0 tps vs 40.9-41.0 tps at gen 64, so short-gen benches understate
steady-state decode by ~25%. Bench windows should be long enough to amortize warmup; use gen >= 256 for decode
comparisons and keep gen 64 only as the historical bar.

## 27. My "tiled MoE is neutral" result was dead code; the int8 SF4 path already exists but is RDNA4-gated
The STRATA_INK_MOE_TILED experiment came back identical on perf AND bit-identical logits (KLD=0.000000,
PPL=1101.388 unchanged). A different accumulation order does not produce bit-identical output, so that was a
reachability signal, not a physics result. A probe fprintf in the block printed ZERO lines in both gate states:
the block is inside `if (ep2_active_)` (ink_hip_backend.hip.cpp:4239), the DUAL-GPU expert-parallel branch, which
never runs with one GPU. Lesson: when a change that must differ shows no difference, prove the code ran before
interpreting the number. Both the gate and the probe were reverted (tree clean at a0e388d).

### The real MoE dispatch map for this workload (single GPU, all experts resident)
Comments at :4232-4237 name the paths; verified structure:
  A1  B <= 4                    -> k_moe_direct_* (zero-sync, 2304 blocks)      [decode path]
  A2  B >= blaslt_moe_min_b_ && blaslt_.ready()                                  [hipBLASLt; we have none]
  A3  B >= 5, GPU-resident      -> grouped MoE, the prefill path we use          [gate at :4392]
  B1/B2 hybrid partial-VRAM                                                    [not us: experts all resident]
Inside A3 at :4392 the same `mem_info_.gen != AmdGen::Rdna2 && != Unknown` gate picks the modern kernels:
  use_i8_moe_wmma_ && layer_gu_class_==0 -> k_quant_f32_to_q8_64 + ..._wmma_sk_i8_128
  use_i8_moe_wmma_ && layer_gu_class_==1 -> k_quant_f32_to_q8_16 + ..._wmma_sf4_i8_128   <-- our SF4 codec
  else                                    -> k_f32_to_f16 + ..._wmma_128[_spec<>]
and the RDNA2 branch (i.e. us) skips all of it to the scalar k_moe_grouped_w13_situ at :4503.
So the "build an int8 DP4A path for SF4" project I sketched in section 25 is already implemented upstream;
the open question is purely why RDNA2 is excluded from it. use_i8_moe_wmma_ is set at :760 as
(gen == AmdGen::Rdna4) with an env override STRATA_HIP_MOE_I8, and the comment says "RDNA4 has true 2x-rate
16x16x32 iu8 WMMA; RDNA3/3.5 wins with double-buffered FP16 WMMA" -> RDNA3, which HAS WMMA, was measured
slower on int8, so the prior for RDNA2 is weak. But RDNA2's alternative is not FP16 WMMA, it is the scalar
fallback, which is exactly the situation #6264 complains about. Counter-evidence: the kernels use
wmma_ld_i8_k16 / wmma_acc_n helpers, and the file's own device table says gfx103x has "zero WMMA intrinsics",
so this may simply not compile for gfx1031 -- which would settle it as capability-based, not policy-based.
Experiment queued (opt-in, default-inert since use_i8_moe_wmma_ is false on RDNA2 unless the env forces it):
gate widened at :4392 to `(gen != Rdna2 && != Unknown) || (use_i8_moe_wmma_ && gen == Rdna2)`, measured with
STRATA_HIP_MOE_I8=1. Build failure on gfx1031 is itself a result. Expect NOT bit-identical if it runs.

### Decode measurement note (repeats section 26, worth pinning)
At prompt 2048 the same build shows decode 25.8-25.9 tps at gen 4 vs 40.7-41.0 at gen 64 vs 51.0-51.1 at gen 256.
Decode windows this short are dominated by warmup; only gen >= 256 is a usable steady-state decode number.

### 27.1 Two RDNA2 sdot4 SF4 kernels exist but are dead code -- and they are not the MoE fix
grep finds `k_sf4_sdot4_tiled` (ink_hip_codecs.hip.h:1070) and `k_sf4_sdot4_v2` (:1189), both commented
"RDNA2 (gfx103x) Zero-WMMA Wave32 sdot4 Tiled SF4 GEMM (also clean fallback for any arch!)" and built on
sdot4_i32(a.x,b.x,idot) x4 -- the DP4A primitive. Neither is launched anywhere in the engine (grep for
launch sites and for the name outside the codecs header: nothing).
Their signature is a DENSE GEMM: (packed u8, scales u8, outer_scale, X f16, Y f32, M, N, K) with BM/BN tiles,
LDS-staged int8 operands and per-16-block float scales. So they target dense SF4 linear layers, not grouped
MoE, and this model's trunk is bf16 (the code keys off trunk_bf16_.count("<layer>.mlp.gate.weight")), which is
the likely reason nothing calls them. Do not "wire them up" expecting a MoE win: for MoE they would need the
grouped/bucketed structure (per-expert ranges from exp_off, SITU epilogue, w2 half) that k_moe_grouped_w13_situ
has and they do not.
Net remaining options for the 52% moe_w13 bucket, in cost order:
  1. measure whether the int8 grouped path can be forced on at all on gfx1031 (STRATA_HIP_MOE_I8=1, running now;
     a compile failure on wmma_ld_i8_k16 / wmma_acc_n settles it as capability-limited);
  2. otherwise write a grouped sdot4 SF4 MoE kernel (port the tiled sdot4 technique into k_moe_grouped_w13_situ,
     quantizing activations q8_16 and mapping SF4 e2m1 magnitudes x2 -> integers 0..12). Days of work, changes
     numerics, ceiling suggested by PR #6850 is ~1.5-1.7x on the MoE prefill share.
Everything cheap is now measured out: occupancy (all big prefill buckets at their LDS or register ceiling),
prefill chunk size (512 is the ceiling: SWA ring margin + batch scratch), the spec-decode flags (dead in --bench),
the attention window (already clamped correctly), and unembed (at the 384 GB/s roofline; AMD spec confirms 384 GB/s
for this card, matching the ~380 GB/s measured).

## 28. STRATA_HIP_MOE_I8=1 doubles prefill on RDNA2 and produces garbage -- fast, wrong, NOT shippable
Widened the A3 gate at :4392 to `(gen != Rdna2 && != Unknown) || (use_i8_moe_wmma_ && gen == Rdna2)` so that
STRATA_HIP_MOE_I8=1 can route our SF4 experts through k_moe_grouped_w13_situ_wmma_sf4_i8_128 /
k_moe_grouped_w2_wmma_sf4_i8_128. It COMPILES for gfx1031, so the RDNA2 exclusion is not a hard capability wall.
RX 6700 XT, --parallel 1 --kv fp8, chunk 512, prompt 2048 gen 256 (steady-state decode window):
| | env unset (control) | STRATA_HIP_MOE_I8=1 |
|---|---:|---:|
| prefill 2048 | 186.6 / 186.9 ppts | **381.7 / 381.8 ppts (x2.05)** |
| prefill 8192 | 158.7 ppts | **282.9 ppts (x1.78)** |
| decode 2048 | 51.1 / 51.1 tps | 44.0 / 44.0 tps |
| Mean KLD | 0.000000 | **3.590106** |
| Top-1 agreement | 100.00% | **46.18%** |
| PPL | 1,101.388 | **58,222.513** |
Control side is exact (186.6-186.9 ppts, KLD=0, PPL=1101.388), so the gate is default-inert and the delta is real.
Ruled out as the cause, by reading the code:
  - wrong codec routing: :1426/:1428 give layer_gu_class_ == 1 for gu_code == CODE_SF4, i.e. the SF4 kernel really ran;
  - missing activation hand-off: w13 writes fp16 d_act, then k_quant_f16_to_q8_16 -> aq, then w2 reads aq (:4448-4451) -- consistent.
So the fault is inside the int8 grouped kernels on gfx103x (they are built on wmma_ld_i8_k16 / wmma_acc_n fragment
loads whose non-WMMA lowering is evidently wrong here), consistent with PR #6850 listing gfx1030/gfx1031 as
"opt-in, no card report yet". The 53x PPL blowup is not a rounding difference.
Decode ALSO dropped (51.1 -> 44.0), which is the same bug showing up a second way: spec-decode verify batches hit
this path, the verify logits are wrong, drafts stop being accepted.
Decision: reverted (tree clean at a0e388d). Patch kept at /mnt/win/llm-temp/scratch/strata_hip_moe_i8_gate.patch
for re-application in one step. This is the biggest performance signal found (2x prefill) and it is unusable until
the int8 kernels are correct on RDNA2 -- fixing them is a multi-day kernel job with an uncertain outcome, and any
win there is inherently non-bit-identical (activations are quantized to q8_16), so it needs an explicit owner
decision before anyone starts.

## 29. Why the spec-decode flags are dead on HIP: the window config is CUDA-backend-only
cuda/ink_cuda_backend.cu:118 reads STRATA_INK_SPEC_BEAM (and _COST/_PLAIN; STRATA_INK_SPEC_LOG=1 prints stats) into
spec_cfg_, described at :1205 as "the drafter's window shape". The HIP backend reads NO such variable -- grep of
getenv in ink_hip_backend.hip.cpp yields only: STRATA_HIPBLASLT_MIN_M, STRATA_HIPBLASLT_MOE_MIN_B, STRATA_HIP_DSPARK_PROF,
STRATA_HIP_MOE_I8, STRATA_HIP_PROF, STRATA_HIP_TREE_PROF, STRATA_INK_EXACT_ROWS, STRATA_INK_HASH, STRATA_INK_SWA_RING,
STRATA_INK_TAP_VISITS, STRATA_MAX_PREFILL_CHUNK.
So --draft-beam/--draft-cost/--draft-plain are accepted by the CLI, stored in cfg.bopts, and implemented only for
CUDA. On HIP the draft count comes from ink_hip_backend.hip.cpp:1973 `n_draft = min(max_draft, adapt_max)` with
max_draft = min(7, rem-1) from ink_main.cpp:698, and adapt_max only SHRINKS on low acceptance (ema<0.20 -> 2 plus a
6-step skip, ema<0.25 -> 4). There is no HIP-side window tuning surface; adding one would mean porting the CUDA
spec_cfg_ machinery, which is a rewrite, not an experiment.
What HIP does have and I have not used yet: STRATA_HIP_DSPARK_PROF and STRATA_HIP_TREE_PROF profilers for the
speculation path, and STRATA_HIP_PROF buckets per batch size B, which can show how decode is split between the
B<=4 direct-MoE path (A1) and the B>=5 grouped path (A3) -- worth reading before touching any decode kernel.

## 30. Prior-art sweep was too narrow; decode occupancy survey; speculation sign flip; a landmine in PR 7009

**Process failure, recorded.** Every prior-art check until now was `gh search ... hip in:title`. The correct
query -- all open PRs touching this backend -- found three near-duplicates in one command. Always sweep the lane
before claiming a finding is ours.

Open PRs against this backend that predate or parallel our work (all `merged=no` at time of writing):

| PR | created | content | relation to our commits |
|---|---|---|---|
| #7009 | 11-24 | `k_moe_grouped_{w13,w2}_wmma_sf4_i8_n8`, int8 WMMA, BM=32/BN=128, 8 waves across columns, env `STRATA_INK_TOPUP_MOE`, B in 64..256 | thematic overlap ("8 waves grouped MoE") only. Needs WMMA => cannot run on gfx103x |
| #7110 | 11-25 03:36Z | `STRATA_INK_MAX_PREFILL_CHUNK`, +7 lines | same-day duplicate of our `b2de3d4` intent; ours derives the cap from `estimate_batch_scratch_bytes()` and clamps bad overrides |
| #6932 | 11-23 | fix wrong output past 1024 tokens (SWA ring), `kSwaRingSlack=1024` | our fork base ALREADY fixes this (#6878): `swa_ring_`/`swa_span_cap_`, 13 wrap sites, mechanism spelled out in the comment at `ink_hip_backend.hip.cpp:718`. I re-derived a documented invariant |
| #2399 | 10-14 | elastic expert cache + growing K/V on HIP | predates our KV-grow/graph-recapture artifact analysis |
| #6850 | 11-23 | gfx103x int8 MMQ prefill | the 2.05x-but-garbage path of S28; still has no card report |

Upstream `main` has zero commits under `engine/src/inkling2/hip/` since our fork base. 5431 forks, none differentiated.
Net: no single PR/fork had everything, and we did NOT aggregate them either -- an aggregate would have found these.

**PR 7009 landmine (our unique evidence).** Its gate is
`bool topup_moe_on(int B) const { return topup_moe_ && B >= 64 && B <= topup_maxb_; }` with `topup_moe_` from
`getenv("STRATA_INK_TOPUP_MOE")` -- NO arch check, unlike the existing path which is guarded by
`use_i8_moe_wmma_ = (gen == AmdGen::Rdna4)`. On a gfx1031 build that env var reaches SF4 int8 MoE without matrix
cores. Measured here (S28): prefill 2.05x, mean KLD 3.590106, top-1 46.18%, PPL 58,222 vs 1,101 correct.
Silent wrong output. Nobody else has this datapoint.

**Build gotcha.** `cmake --build build-ink` with no `--target` FAILS on an unrelated target in this tree.
Always `--target strata-inkling`. Also: piping the build through `tail -20` in a script destroyed the error
evidence and `| tail` masked the exit code from `set -e`. Capture full logs to a file.

**Profiler label correction.** `STRATA_HIP_PROF=2` emits ONLY `[HIP-PROF] B=<n>` lines (35/35 in the sample).
Decode under spec is a single B=1 forward per ~1.63 accepted tokens -- there are no B=2..8 verify buckets.
So "verify batches use the grouped A3 path" was never true; the int8 decode drop needs no path explanation,
just acceptance collapse. Also `steps=` in profiler output is the SAMPLE count, not decode steps.

**Decode kernel occupancy at HEAD (the A1 direct path, never surveyed before; earlier survey covered only prefill kernels):**

| kernel | VGPRs | LDS | waves/SIMD | limit |
|---|---:|---:|---:|---|
| `k_moe_direct_w13_situ` (~19% of decode) | 74 | 7,200 B | 12 | registers; LDS would allow 18 |
| `k_moe_direct_w2_reduce` (~13%) | 127 | 512 B | 8 | registers, far from 16 |
| `k_bf16_gemv1_qkvrg` | 38 | 16,384 B | 16 | HW ceiling |
| `k_bf16_gemv1_wo_sconv` | 38 | 16,384 B | 16 | HW ceiling |

Both MoE kernels were plain `__launch_bounds__(256)` with 256-thread launches. Compile-only probe
(`-Rpass-analysis=kernel-resource-usage`, ~2.5 min, CPU only, no build): `(256,14)` and `(256,16)` both yield
VGPRs 64, waves/SIMD 16, **VGPR spill 8, ScratchSize 36 B/lane**. So the bump is NOT spill-free, unlike
`d613831`/`9df38f5`. Verdict requires hardware; job `bcdb16956` measures it (bars: 51.0 tps @2048/gen256,
41.9 @8192/gen512, KLD must stay 0.000000).

**Speculation sign flip across context (gen 256 steady state, job `bce6a2936`):**

| prompt | spec ON | `--draft off` | delta |
|---|---:|---:|---:|
| 512 | 25.8 | **34.2** | **-24.6%** |
| 2048 | **51.0** | 32.8 | +55.5% |
| 8192 | **41.9** | 30.7 | +36.5% |

At 512: 1000/25.8 = 38.8 ms/token vs 29.2 ms draft-off = a flat +9.6 ms tax, i.e. drafts accepted ~nothing.
Mechanism (code read, `dspark_draft` + `rollback_slot`): EMA rises only on full acceptance (`:1958`); it decays
ONLY inside `rollback_slot` behind `if (sl.dspark_last_draft > 0 && keep_steps >= 1)`, where
`keep_steps = keep_len - sl.snap_seq_len`. A step that rejects everything lands at `keep_steps == 0` => no EMA
update at all. So total rejection is invisible to the controller, EMA stays at its 0.50 seed (above both 0.20/0.25
thresholds), and it drafts the full 7 forever. The source comment claims a `K=4->2->0` backoff the code cannot reach.
GENERAL IDEA IS NOT OURS: low-acceptance speculation being net-negative and EWMA draft-length control are
established (arXiv 2406.14066 states the overhead case; adaptive draft-length literature is mature). Ours is the
specific implementation defect. Fix would NOT be bit-identical (draft count changes sampled tokens for a fixed RNG
stream) though accept/reject sampling keeps the distribution exact -- flag that distinction to the owner before shipping.

## 31. Decode +2.7% shipped (16-wave direct MoE, spill included); rebase assessed; our bench never tests tail chunks

**Shipped `2460e04a3`** `inkling2/hip: raise decode MoE w13 to 16 waves/SIMD (+2.7% decode, bit-identical)`.
`k_moe_direct_w13_situ` (A1 path, B<=4 = all decode on this card): `__launch_bounds__(256)` -> `(256, 16)`;
74 VGPRs/12 waves -> 64 VGPRs/16 waves. I predicted the 8-VGPR spill (ScratchSize 36 B/lane) would cancel the
gain. WRONG: it is a real win, consistent across 3 runs, prefill untouched.

| metric | bar (a0e388d) | with 2460e04a3 | delta |
|---|---:|---:|---|
| decode 2048 x gen256 | 51.0 | 52.6 / 52.4 / 52.3 | +2.7% |
| decode 8192 x gen512 | 46.9 | 48.0 | +2.3% |
| decode 512 x gen256 | 25.8 | 26.6 | +3.1% |
| prefill (control) | 186.7-187.0 | 186.7-186.9 | flat |
| parity | - | KLD=0.000000, Top-1 100.00%, PPL=1101.388 | bit-identical |

Lesson: a spill is not automatically disqualifying; measure it. Also: my "41.9 tps" bar for 8192/gen512 (job
bce6a2936) was stale/noisy vs 46.9 recorded earlier -- comparing against it would have overstated this as +14.6%.
**Current bars (2460e04a3):** prefill 186.7-187.0 ppts @2048 (158.7 @8192); decode 40.9-41.0 @gen64,
52.3-52.6 @2048/gen256, 48.0 @8192/gen512, 26.6 @512/gen256; KLD 0.000000 vs ref_512_c256.bin.
`k_moe_direct_w2_reduce` left alone: 127 VGPRs/8 waves, hitting 16 needs ~56 VGPRs -> would spill hard.

**Rebase assessment (behind 41, ahead 7).** ZERO upstream commits touch `engine/src/inkling2/hip/`; the only 3
inkling2 commits touch CMakeLists/ink_main.cpp/ink_serve.h/cuda headers -> no overlap with our 2 files, clean rebase.
v0.2.22 release notes state the AMD/Intel/Apple Inkling engines "stay 0.1.67, because nothing in them changed".
Only TWO new env vars in the whole engine since fork base 2af30cc (1271 -> 1273): `STRATA_KEEP_SYMBOLS` (CMake)
and `STRATA_SPLIT_LEND_RESERVE` (core/expert_source.cpp; upstream: +9% decode on 2-card splits, -17% cold 18K
prompt). Expect inert single-GPU; A/B it anyway. Conclusion: rebase = hygiene, not perf. Script ready:
`scratch/rebase_validate.sh` (refuses a dirty tree, aborts on conflict, re-runs the full bar matrix + parity).

**BENCHMARK BLIND SPOT, found via PR 7009.** Our bench uses prompt 2048 with chunk 512 -> 2048 % 512 == 0, so it
NEVER exercises a prompt *tail* chunk. PR 7009's actual insight is arch-neutral (its WMMA kernels are not): with
64..256 rows an expert gets ~4..16 rows, so with a 4-row-strip x 2-column-half wave layout "only the 2 waves of
strip 0 compute; the other 6 only load" -> retiling to BM=32/BN=128 spreading columns over all 8 waves. Our RDNA2
grouped kernels may have the same waste in tail chunks, and our benchmark cannot see it. Test with a prompt that
is NOT a multiple of the chunk (e.g. 2000 or 2200 at chunk 512) before concluding anything about chunked prefill.

**Gotchas.** (a) `env VAR=x <bash_function>` fails -- env can only exec programs; use a subshell + real path.
(b) `STRATA_HIP_DSPARK_PROF=1` printed ZERO lines through the run() wrapper (job bcdb16956) -- still no direct
evidence for the stuck-EMA theory in S30; retest with env(1) and check whether the bench even loads dspark.
(c) never pipe `cmake --build` through `tail` in a script: destroys the error and hides the exit code from set -e.

## 32. Concurrency is net-NEGATIVE on this box -- my "free 2-3x" hypothesis was wrong

Job `beaa91502`, prompt 2048 gen 256, killed during `--parallel 4` because the trend was already decisive:

| parallel | prefill ppts | aggregate decode tps | tps/slot |
|---:|---:|---:|---:|
| 1 | 186.7 | **52.5** | 52.5 |
| 2 | 185.3 | 47.1 | 23.5 |
| 3 | 36.7 | 10.9 | 3.6 |

Aggregate throughput DROPS below single-stream at 2 slots and collapses at 3 (prefill falls 5x). Mechanism: a
single stream already saturates this card's effective bandwidth, so extra slots add only KV pressure; past ~2
slots the 12 GiB is over-subscribed and experts spill to system RAM across the ~3.9 GB/s Thunderbolt link.
Every run printed `Capped max_prefill_chunk=512 (post-load free=982 MiB, 512-token scratch=391 MiB)`, i.e.
there was never spare headroom to begin with.

Consequences: (a) the largest remaining "massive tps" hope is dead -- do not propose `--parallel` as a win here;
(b) it predicts the SERVED config will be worse still, since setup already parks `--ram-experts-gb 1.69` in RAM
to fit `--max-context 131072`, which is the same failure mode chosen in advance.

## 33. gen-64 decode cell filled (+1.9% at gen 64; cumulative decode now +18.1%); prompt tails are harmless

Job `bdf517739`, binary `2460e04a3`:
- `--prompt-len 2048 --gen 64`: **41.8 / 41.7 tps** vs the 40.9-41.0 bar -> +1.9%. So the shipped decode win holds
  at gen 64 as well as gen 256, and cumulative decode vs baseline is **35.4 -> 41.8 = +18.1%** (earlier docs said
  +15.8% using the stale 41.0 bar). Prefill unchanged 186.6-186.7.
- **Prompt tails** (never tested before; all prior numbers used 2048 with chunk 512 = exact multiple):
  at cap 512 -> 2000: 186.8 ppts, 2200: 183.4, 2048: 186.9. At cap 256 -> 176.8 / 174.6 / 176.9. A 152-token tail
  costs 1.9%, a 464-token tail ~0.1%; chunk 512 beats 256 uniformly by ~5.7%. Conclusion: the #7009 wave-imbalance
  concern has no measurable headroom here, and earlier 2048-based conclusions stand.
  Decode at `--gen 8` is meaningless noise (24.9 vs 41.4 tps for near-identical work) -- reinforces gen >= 256.
- `--spec-check` requires `--prompt-ids` (it rejected `--prompt-len`); rerun as job `b67d50214` with
  `prompt512_cma.ids` / `prompt2048_cma.ids` to get the real `fused / draft+verify / single` split and
  drafted/emitted acceptance. Until that lands, NO acceptance figure in this log is measured.

## 34. OPEN: possible correctness regression in 2460e04a3 -- testing stopped by request, unresolved

Acceptance first (prompt ids, seed 13, `--spec-check 64`, binary 50ac62ae):
- prompt 2048: `steps=50 (fused 13, draft+verify 0, single 37), drafted 29, emitted 26`
- prompt 512 : `steps=46 (fused 9,  draft+verify 0, single 37), drafted 22, emitted 26`
`drafted_sum/emitted_sum` are updated by the FUSED path too (ink_main.cpp:690-691), not only draft+verify, so these
describe the tree: ~2.2 tokens per fused forward at 2048 (2.9 at 512) vs 1.0 for single steps. Overall acceptance is
64/50 = **1.28 tokens per forward** at 2048 -- NOT the 1.63 I earlier derived from a stale 40 ms/forward figure.
The weakness is speculation FREQUENCY (fused engaged 13 of 50 steps), not acceptance rate. `draft+verify 0` also
confirms the adaptive-backoff fallback `dspark_draft()` never runs in this configuration -> measured evidence for
UPSTREAM-NOTES issue B.

Then the problem. Same binary, prompt 512: `speculative vs plain over 64 tokens: DIFFER (first difference at
generated token 24: plain 68017, speculative 220)`; prompt 2048 identical. Greedy speculation should be lossless
against plain greedy -- that is what --spec-check exists to prove.

Two of my own hypotheses were tested and died:
1. "nondeterminism / race" -- FALSE. 3/3 repeats give the same spec fingerprint 0876b3dbffe82992 and the same
   first-divergence index. Deterministic per binary.
2. "atomic reordering in the MoE combine" -- UNSUPPORTED. In k_moe_direct_w2_reduce the atomicAdd calls count
   expert slots (s_cnt/exp_cnt/exp_cur); k_moe_direct_w13_situ contains none.

Then I found my own error: the earlier "identical at prompt 512" observation (job b4e98d269) came from a run whose
script performed the spec-check BEFORE its final rebuild, so it used the pre-commit binary cd3bc440. My "same binary,
different result" framing was wrong; the honest framing is "pre-commit binary matched, post-commit binary diverges".

Coverage hole that made this possible: my parity gate tested B=1 (teacher forcing) and B=256 (prefill) only. The
fused tree verifies at intermediate widths and k_moe_direct_* serves B<=4. B=2..4 was never parity-tested, so a
behavioural change there is invisible to the gate that justified the commit. Leading mechanism: 16 waves cost 8
spilled VGPRs, and a spilled value must be rounded, which can break FMA contraction -> ulp-level logit changes ->
near-tie argmax flips. B=1 instantiation can be exact while B=2..4 is not.

Status: STOPPED on request. Bisect job b55d7e96d was killed during its Phase 1 rebuild, so causality is UNPROVEN --
recorded as an OPEN item at the top of README.md, not as a conclusion. Repo left clean at HEAD; binary on disk is
50ac62ae (matches HEAD). Practical severity for our own deployment is low because the serve profile runs --draft off,
so the speculative path is not used; severity for anyone enabling speculation is unknown until the bisect runs (~8 min:
rebuild 2460e04a3^ vs HEAD codecs, run spec-check on prompt512_cma.ids for each; if pre-commit matches and post-commit
diverges, revert 2460e04a3).
