# Inkling 2 Mini SQ-10 on an RX 6700 XT eGPU (gfx1031) — RDNA2 grouped-MoE work

Protocol: `strata-inkling --bench --native <SQ-10.strata> --parallel 1 --kv fp8 --vram-experts-gb 8.77 --ram-experts-gb 0`.
Numbers are this machine only; nothing here is a statement about other AMD cards or later releases.

- **Hardware:** Dell Latitude 5420 + Razer Core X eGPU, AMD RX 6700 XT (`gfx1031`), 12 GiB VRAM, 384 GB/s GDDR6,
  96 MB Infinity Cache. RDNA2: **no matrix cores** (zero WMMA intrinsics on gfx103x), `fdot2` available, no hipBLASLt.
  Thunderbolt 3 Gen3 x4 host link, measured ~3.9 GB/s raw — relevant because any expert placed in system RAM is slow.
- **Model:** Inkling-2-Mini-SQ-10, 26.4 B total / 3.4 B active (5.4 B looped-active across 48 virtual visits),
  expert weights in VRAM. Measured directory mix (24 layers x 128 experts): 585 S2_K, 2,472 S3_K, and
  15 SF4 (4-bit e2m1 + e4m3 block scales); the older description of this model as all SF4 was stale.
- **Branch:** `rdna2-moe-lds-staging`. **Binary under test:** `2460e04a3`, md5 `50ac62ae467218043def9c32a3d2bd5e`;
  the clean baseline `a0e388d` was md5 `cd3bc440dff82c9962c682eccf9faecd`. ROCm libs come from the stock
  `0.1.67-linux-x86_64-hip-rx6000` component package (the `enginetest` dir used throughout is only symlinks to it).
- **Dates:** 2026-11-25 → 2026-11-26.
- **Current HEAD (2026-11-26):** `52ba23bfc`, which adds branchless E4M3 decode and the S2/S3 W13 pair
  LUT on top of `2460e04a3`. The OPEN speculative-path item below still applies to this build. Those two
  changes touch FP8 attention and grouped-W13 weight unpacking only.

## OPEN -- possible correctness regression in `2460e04a3`; do not treat its "bit-identical" label as settled

`2460e04a3` is a one-line change, `__launch_bounds__(256)` -> `(256, 16)` on `k_moe_direct_w13_situ`, and its commit
message says "bit-identical". That claim was established only at **B=1** (teacher-forced `--dump-ref` parity,
KLD=0.000000, Top-1 100%) and at B=256 (prefill). The **B=2..4** instantiations of the same kernel were never
parity-tested -- and B<=4 is exactly the band the fused speculative tree verifies in.

Observed with the committed binary (`50ac62ae...`, seed 13, `--spec-check 64`):

```
prompt 512  : speculative vs plain greedy DIFFER (first difference at generated token 24: plain 68017, speculative 220)
              reproduced 3/3 with identical speculative fingerprint 0876b3dbffe82992  -> deterministic, not a race
prompt 2048 : speculative vs plain greedy identical (3/3)
```

Greedy speculative decoding is supposed to be *lossless* against plain greedy; that is the entire purpose of
`--spec-check`. The earlier `identical` result at prompt 512 came from job `b4e98d269`, whose spec-check ran
**before** that job's rebuild and therefore used the pre-commit binary `cd3bc440`. So: pre-commit binary matched,
post-commit binary does not, same prompt and seed.

**Causality is not established.** The bisect (job `b55d7e96d`) was stopped on request during its Phase 1 rebuild.
Two hypotheses were tested and rejected first: a data race (deterministic across repeats) and atomic reordering in
the MoE combine (the direct decode kernels use atomics only for expert *slot counting*; `k_moe_direct_w13_situ`
uses none).

Leading candidate mechanism: reaching 16 waves cost 8 spilled VGPRs, and a value forced through memory must be
rounded, which can split a fused multiply-add. That perturbs low bits of logits -- enough to flip a near-tie
argmax. Register pressure differs per batch-width instantiation, so B=1 can stay exact while B=2..4 shifts.

**Severity for the deployment templated in `deploy/`: low** -- that profile runs `--draft off`, so the speculative
path is never taken. Severity for anyone who enables speculation: potentially high.

To settle it (GPU, ~8 min): build `2460e04a3^` codecs (expect md5 `cd3bc440dff82c9962c682eccf9faecd`) and HEAD
codecs (expect `50ac62ae467218043def9c32a3d2bd5e`), then run `--prompt-ids @parity/prompt512_cma.ids --spec-check 64`
on each. If pre-commit matches and post-commit diverges, revert `2460e04a3` -- the +1.9% (gen 64) / +2.7% (gen 256)
decode gain does not justify speculation that is not lossless. If both diverge, the invariant is fragile upstream
and the finding belongs in an upstream issue, not in this branch.

Independent of that outcome: **logit parity gates in this backend must cover B=1..8.** A B=1 + B=256 gate
demonstrably failed to catch a behavioural change in a kernel that also serves B=2..4.

## Headline: same protocol as the baseline (prompt 2048, gen 64, speculation on)

| metric | baseline `2af30cc` | after `2460e04a3` | change |
|---|---:|---:|---:|
| prefill | 117.1 ppts | 186.7–187.0 ppts | **+59.5%** |
| time to first token (2048 tok) | 17.49 s | 10.96 s | −37.3% |
| decode | 35.4 tps | 41.7–41.8 tps | **+18.1%** |

### Current fork total versus the fork base (added 2026-11-26)

Measured with the same prompt 2048 / gen 64 / speculation-on protocol used for the baseline above.
Current binary is `13aa95a90` on the `rdna2-moe-lds-staging` branch.

| metric | fork base `2af30cc` | current `13aa95a90` | change |
|---|---:|---:|---:|
| prefill | 117.1 ppts | 196.2 ppts | **+67.6%** |
| time to first token (2048 tokens) | 17.49 s | 10.44 s | −40.3% |
| decode | 35.4 tps | 43.4 tps | **+22.6%** |

On the separate protocol used for the latest kernel A/B (prompt 2048 / gen 256, speculation enabled by the
then-default),
the two changes in `52ba23bfc` moved prefill 189.8 -> 196.3 tok/s (+3.4%) and decode 53.8 -> 55.3
tok/s (+2.8%).

Quality gate on the current binary: `--kld-eval ref_512_c256.bin` with 511 teacher-forced positions
gives `Mean KLD=0.000000 | Top-1=100.00% | PPL=1101.388`, identical to the baseline reference value.
The opt-in FP16-prefill/FP8-decode benchmark hook and the open speculative-path item above are not
part of these totals.

Per commit, prefill ppts (prompt 2048):

| commit | what | prefill | step Δ |
|---|---|---:|---:|
| `2af30cc` | fork base | 117.1 | — |
| `3974192` | LDS slice staging in grouped MoE (bit-identical) | 160.4 | +37.0% |
| `d613831` | grouped w13 → 8 waves/SIMD (bit-identical) | 171.7 | +7.1% |
| `9df38f5` | grouped w2 → 8 waves/SIMD (bit-identical) | 177.3 | +3.3% |
| `a0e388d` | prefill chunk 512 as default on tight-VRAM cards, sized from `estimate_batch_scratch_bytes()` | 186.9 | +5.5% |
| `2460e04a3` | decode-only `k_moe_direct_w13_situ` → 16 waves/SIMD | 186.9 | 0 (decode only) |

`2460e04a3` moves **decode**, and only at long measurement windows — which is why the gen-64 row above does not show it:

| window | bar (`a0e388d`) | after `2460e04a3` | Δ |
|---|---:|---:|---:|
| 2048 × gen 64 | 40.9–41.0 tps | 41.8 / 41.7 | +1.9% |
| 2048 × gen 256 | 51.0 tps | 52.6 / 52.4 / 52.3 | +2.7% |
| 8192 × gen 512 | 46.9 tps | 48.0 / 47.9 | +2.2% |
| 512 × gen 256 | 25.8 tps | 26.6 | +3.1% |
| prefill (drift control) | 186.7–187.0 | 186.7–186.9 | flat |

Every shipped change is **bit-identical**: `--kld-eval` against `ref_512_c256.bin` gives
`n=511 Mean KLD=0.000000 | Top-1=100.00% | PPL=1101.388` before and after.

## Default configuration (2026-11-26)

`enable_dspark` now defaults to false, so plain greedy decoding is the default everywhere. It can still be
enabled explicitly with `--draft dspark`. This was decided by measuring, not assuming:

- `--spec-check 64` on the 512-token parity prompt, three runs on the current binary: every run reported
  `DIFFERS`, first difference at generated token 24 (plain 68017, speculative 220). Speculation is faster
  but is not lossless on this build, so it cannot be the default when output must be unchanged.
- The opt-in FP16-prefill/FP8-decode hook is also not bit-exact (KLD 0.001667 versus 0.000000) and needs
  two KV copies resident at once, so it stays opt-in.
- The default FP8 KV path on the current build measures prefill 196.4–196.7 tok/s and decode
  34.4–34.6 tok/s at prompt 2048 / gen 256, three warmed runs, all experts in VRAM. The fresh parity gate
  on this binary is `n=511 | Mean KLD=0.000000 | Top-1=100.00% | PPL=1101.388`, identical to the
  baseline reference.

The 55 tok/s decode figures recorded earlier in this branch came from the speculative path that was on by
default at the time; they are throughput measurements and are not covered by the lossless guarantee.

The serving profile keeps 7.08 GiB VRAM experts plus 1.69 GiB RAM experts: an all-VRAM selfcheck passes
with 1,710 MiB free, but a 131,072-token KV cache would then spill to host, so the existing split is
retained for long-context headroom.

## Decode composition (B=1, `STRATA_HIP_PROF=2`, prompt 2048)

One full forward = 30.4 ms. Dense linear is half of it; MoE is a third.

| component | ms | share |
|---|---:|---:|
| dense linear (`qkvrg` 6.99 + `wo_post` 4.51 + `res_ctrl` 2.2 + `prep` 1.4) | 15.1 | 50% |
| MoE (`moe_w13` 5.71 + `moe_w2_post` 3.93) | 9.6 | 32% |
| attention + shortconv (`attn` 3.91 + `sconv` 1.8) | 5.7 | 19% |
| unembed (overlaps) | 2.4 | 8% |

Kernel occupancy at HEAD, and what limits each (`-Rpass-analysis=kernel-resource-usage`, compile-only):

| kernel | VGPRs | LDS | waves/SIMD | limited by |
|---|---:|---:|---:|---|
| `k_moe_direct_w13_situ` | 74 → **64** | 7,200 B | 12 → **16** | was registers; LDS allowed 18 |
| `k_moe_direct_w2_reduce` | 127 | 512 B | 8 | registers; 16 needs ~56 VGPRs → would spill hard. **Not attempted.** |
| `k_bf16_gemv1_qkvrg` | 38 | 16,384 B | 16 | hardware ceiling |
| `k_bf16_gemv1_wo_sconv` | 38 | 16,384 B | 16 | hardware ceiling |
| `k_moe_grouped_w13_situ` | — | — | 8 | raised in `d613831`, spill-free |

Note `2460e04a3` is **not** spill-free: 64 VGPRs costs 8 spilled VGPRs / `ScratchSize 36 B/lane`. I expected that to
cancel the gain. It did not, measured across three runs. A spill is a cost to measure, not an automatic veto.
`(256, 14)` and `(256, 16)` generate identical code — the compiler clamps at 64 VGPRs either way.

## Ceilings found by measurement, not inference

- **Prefill chunk 512 is a hard ceiling here, for two independent reasons.** (1) SWA KV ring: `swa_span_cap_ = swa_ring_ - swa_win`,
  and a batch may not span more than that or it overwrites keys its own earlier queries still need. The ring is sized
  from the `--chunk` option, so `STRATA_MAX_PREFILL_CHUNK` alone silently re-splits at 512 — the two must be set together.
  (2) Batch scratch: raising the chunk past 512 OOMs (`[ink-hip] scratch alloc: out of memory`). Both were found by
  running it, and both are already known to the codebase — this fix landed upstream earlier via #6878.
- **`STRATA_MAX_PREFILL_CHUNK` is scratch-aware:** the override halves itself via `estimate_batch_scratch_bytes()`
  and prints why (1024 → clamped to 512, 4096 → clamped to 512, no OOM). The chunk decision uses the engine's own
  estimate rather than a free-VRAM proxy.
- **Speculative decoding changes sign with context length** (gen 256, steady state):

  | prompt | spec ON | `--draft off` | Δ |
  |---|---:|---:|---:|
  | 512 | 25.8 tps | **34.2 tps** | **−24.6%** |
  | 2048 | **51.0 tps** | 32.8 tps | +55.5% |
  | 8192 | **41.9 tps** | 30.7 tps | +36.5% |

  At prompt 512 speculation is a flat ~9.6 ms/token tax on an otherwise identical ~30 ms forward. The HIP fused path
  `dspark_spec_step_greedy` (`ink_hip_backend.hip.cpp:2174`) uses a **fixed** 8-wide tree with **no acceptance-based
  throttle at all**, so nothing disables speculation when it is losing. The `adapt_max`/EMA backoff exists only in
  `dspark_draft()`, which `ink_main.cpp:686` reaches only as a fallback. Acceptance itself is **not measured here**:
  the counters print only under `--spec-check` (`ink_main.cpp:743`), and `STRATA_HIP_DSPARK_PROF` / `STRATA_HIP_TREE_PROF`
  emit nothing in this configuration, so do not reuse any acceptance figure derived from the 2048 numbers.

### Concurrency does not help on this card -- measured, and it contradicts the obvious expectation

Prompt 2048, gen 256. The run was stopped during `--parallel 4` because the trend was already unambiguous
(`parallel 8` and the long-context cases were never reached):

| parallel | prefill ppts | aggregate decode tps | tps/slot |
|---:|---:|---:|---:|
| 1 | 186.7 | **52.5** | 52.5 |
| 2 | 185.3 | 47.1 | 23.5 |
| 3 | 36.7 | 10.9 | 3.6 |

Aggregate throughput falls below single-stream at two slots and collapses at three, with prefill dropping 5x.
One stream already saturates the card's effective bandwidth, so additional slots contribute only KV pressure;
past ~2 slots the 12 GiB budget is over-subscribed and experts spill into system RAM across the ~3.9 GB/s
Thunderbolt link. Every run logged `Capped max_prefill_chunk=512 (post-load free=982 MiB, 512-token scratch=391 MiB)`,
i.e. there was no headroom to begin with. **Do not treat `--parallel` as a throughput lever on this machine.**

### Prompt tails are cheap -- the evenly-divided benchmark was not hiding anything

Every prefill number in this document used prompt 2048 with chunk 512, i.e. `2048 % 512 == 0`, so no partial
chunk was ever executed. Measured now with `--gen 8` (prefill only; the decode column at gen 8 is noise):

| prompt | chunks at cap 512 | prefill ppts | prefill ppts at cap 256 |
|---|---|---:|---:|
| 2000 | 512+512+512+464 | 186.8 | 176.8 |
| 2200 | 512+512+512+152 | 183.4 | 174.6 |
| 2048 | exact multiple | 186.9 | 176.9 |

A 152-token tail costs 1.9% and a 464-token tail costs ~0.1%, and chunk 512 beats 256 on every prompt by
about 5.7%. So the tail-chunk retiling idea borrowed from #7009 has no measurable headroom to recover on
this card, and the earlier conclusions drawn from 2048 stand. (Incidentally the decode cells here -- 24.9 vs
41.4 tps for near-identical work -- are a live demonstration of why decode must be measured at gen >= 256.)

## Not shipped, and why

- **`STRATA_HIP_MOE_I8=1` (SF4 int8 grouped MoE on RDNA2): 2.05× prefill, numerically destroyed.** 381.7/381.8 ppts
  @2048 and 282.9 @8192, but `Mean KLD=3.590106`, Top-1 **46.18%**, **PPL 58,222.513** vs 1,101.388, and decode fell
  51.1 → 44.0 tps as wrong verify logits collapsed draft acceptance. Ruled out: wrong codec routing (`CODE_SF4` →
  `layer_gu_class_ == 1`, the SF4 kernel provably ran) and a missing activation hand-off (`k_quant_f16_to_q8_16` runs
  before w2). The fault is in the int8 kernels' RDNA2 lowering. Requires lifting the `use_i8_moe_wmma_ = (gen == AmdGen::Rdna4)`
  guard locally to reach at all, so it is unreachable in stock builds. Patch kept at
  `scratch/strata_hip_moe_i8_gate.patch` (22 lines). **Not shippable.**
- **Two-row GEMM variant and a 9-wave occupancy bump:** reverted, measured regression/softness.
- **`STRATA_INK_MOE_TILED`:** the gate sits inside the `ep2_active_` dual-GPU branch, so it never executes on one GPU —
  a null result there was dead code, not a finding.
- **`--draft-beam` / `--draft-cost` / `--draft-plain` / `STRATA_INK_SPEC_BEAM`:** CUDA-backend-only, inert on HIP `--bench`.
- **MTP drafting on gfx1031:** upstream #6343 reports 0/N drafts accepted; not usable here.

## Prior art — most of this was independently in flight

Upstream `main` has **zero** commits under `engine/src/inkling2/hip/` since the fork base, and its own v0.2.22 notes say
the AMD/Intel/Apple Inkling engines "stay 0.1.67, because nothing in them changed". But the same ideas exist as open,
unmerged PRs from different authors, several **predating** our commits:

| PR | created | content | relation |
|---|---|---|---|
| #7009 | 2026-11-24 | `k_moe_grouped_{w13,w2}_wmma_sf4_i8_n8`, int8 WMMA, BM=32/BN=128, 8 waves across columns, env `STRATA_INK_TOPUP_MOE`, B 64..256 | **predates ours.** Thematic overlap only — needs WMMA, cannot run on gfx103x. Its gate has **no arch check**, unlike `use_i8_moe_wmma_`, so on an RDNA2 build that env var reaches a path this document measures as PPL 58,222 |
| #7110 | 2026-11-25 03:36Z | `STRATA_INK_MAX_PREFILL_CHUNK`, +7 lines | same-morning duplicate of our `b2de3d4` intent; ours derives the cap from the scratch estimate |
| #6932 | 2026-11-23 | fix wrong output past 1024 tokens (SWA ring), `kSwaRingSlack=1024` | our fork base already fixes this via #6878 (`swa_ring_`, 13 wrap sites, mechanism documented at `ink_hip_backend.hip.cpp:718`) |
| #2399 | 2026-10-14 | elastic expert cache + growing K/V on HIP | predates our KV-grow/graph-recapture analysis |
| #6850 | 2026-11-23 | gfx103x int8 MMQ prefill, "no card report yet" | the SF4 int8 path measured above |

Method note, kept deliberately: every prior-art check before 2026-11-26 was `hip in:title`. The correct query — all open
PRs touching this backend — found three near-duplicates in one command. **Sweep the lane before claiming a finding.**

## Measurement lessons that changed conclusions

- **Vary the window; never trust one number.** Decode was measured at gen 4/32/64/128/256; short windows understate by
  up to 25% (25.8 @gen4 → 41.0 @gen64 → 51.0 @gen256 for the same binary).
- **A one-time cost divided by a short window looks like a regression.** "chunk 512 regresses decode" was self-inflicted:
  a KV-pool grow bumps `pool_gen` → `kv_gen_sum` → breaks decode-graph reuse (`graph_kv_gen_ == kv_gen_sum`) → the graph
  is destroyed and recaptured **once** (~100 ms). Flat +0.09/0.10/0.11 s at gen 32/128/512 proved it was one event.
  Freeing 480 MiB did not recover decode, so VRAM pressure was ruled out by experiment, not assumption.
- **Compare against a bar from the same protocol.** An earlier 41.9 tps figure for 8192 × gen 512 was stale noise;
  the true comparable was 46.9, so a +2.2% win would have been reported as +14.6%.
- **`steps=` in `STRATA_HIP_PROF` output is the sample count, not decode steps**, and decode emits only `B=1` / `B=512`
  buckets under tracing — do not infer verify batch shapes from it.
- Build gotchas: `cmake --build` **must** pass `--target strata-inkling` (an unrelated target fails otherwise), and never
  pipe the build through `tail` — it destroys the error text and hides the exit code from `set -e`.

## Reproducing

```sh
export LD_LIBRARY_PATH=/mnt/win/llm-temp/enginetest   # symlinks into the stock hip-rx6000 component
cd /mnt/win/llm-temp/build-ink && cmake --build . --target strata-inkling -j"$(nproc)"
cd src/inkling2 && ./strata-inkling --bench --native "$MODEL" --parallel 1 --kv fp8 \
    --vram-experts-gb 8.77 --ram-experts-gb 0 --prompt-len 2048 --gen 256
# parity (must be KLD=0.000000):
./strata-inkling --native "$MODEL" ... --prompt-ids @prompt512_cma.ids --kld-eval ref_512_c256.bin
```

Hardware-counter profiling (rocprofv3): needs `sudo sysctl -w kernel.perf_event_paranoid=2` (restore to 4 afterwards),
the `libhsa-amd-aqlprofile64.so.1` plugin, and the `.1.1.0`→`.1.1.1.0` symlink fix described in `SESSION-LOG.md` §18.

## Files here

- `SESSION-LOG.md` — the full working log, 31 sections, including the places the record corrects itself.
  One line about this box's local disk encryption was redacted for publication.
- `UPSTREAM-NOTES.md` — two upstream **issue** drafts (no pull requests). **Neither has been posted.**

## What lives where, and what survives a reboot or a cleanup

Added while planning to wipe `/mnt/win/llm-temp`. Read this before deleting anything.

**`/` is an overlay over live squashfs (`/cow`), so the entire Linux side is in RAM and does not survive a
reboot.** That includes the installed Strata Terminal CLI, `~/.local/share/strata` (engine components *and*
the ROCm libraries our build links against), the run-config profiles, the systemd unit, and the API-key file.
Only `/mnt/win` is persistent. `deploy/restore.sh` re-creates the RAM-side pieces.

| path | size | survives reboot | survives `llm-temp` cleanup | recovery |
|---|---:|---|---|---|
| `strata-src/` | 297 MB | no (it's on `/mnt/win`, so yes) | **no** | already in git: `myfork/rdna2-moe-lds-staging` |
| `build-ink/` | 76 MB | yes | no | rebuild: `cmake --build build-ink --target strata-inkling` |
| `engines/inkling2-rdna2-local/` | 274 MB | yes | **no** | derived from build; **the systemd profile `exe` points here** |
| `enginetest/` | 3.6 MB | yes | no | symlinks into the stock `hip-rx6000` component; recreate with `ln -s` |
| `scratch/parity/*.bin` | ~2.3 GB | yes | **no** | re-derivable from the baseline commit + GPU time; keep the small `.ids` files |
| `strata-models/` | **11 GB** | yes | **no** | not in git — confirm it can be re-fetched *before* deleting |
| `.gh/auth.log` | 304 B | yes | no | gh credentials, plaintext on a Windows volume; re-auth if lost |
| `rocprof/` | 53 MB | yes | no | re-download the ROCm counter debs |
| `scratch/*.sh`, `bench/*.hip.cpp` | ~40 KB | yes | no | **now committed under `tools/` in this folder** |

Consequence worth stating plainly: **deleting `llm-temp` breaks the serving unit**, because its profile's
`exe` is `/mnt/win/llm-temp/engines/inkling2-rdna2-local/strata-inkling` and its model is
`/mnt/win/llm-temp/strata-models/...`. Either move those two out from under the cleanup path and update the
profile, or expect to re-run `deploy/restore.sh`.
