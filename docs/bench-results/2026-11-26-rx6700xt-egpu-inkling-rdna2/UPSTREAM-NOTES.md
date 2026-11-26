# Draft upstream reports (ISSUES ONLY -- no PRs, per owner instruction)

Nothing here has been posted. Owner to approve wording before `gh issue create`.

---

## Issue A -- SF4 int8 MoE is numerically wrong on gfx103x, and PR #7009's new path has no arch guard

**Hardware:** Dell Latitude 5420 + Razer Core X eGPU, AMD RX 6700 XT (`gfx1031`), 12 GB VRAM, 384 GB/s GDDR6,
96 MB Infinity Cache, **no matrix cores** (RDNA2: zero WMMA intrinsics; `fdot2`, no hipBLASLt).
Model: Inkling-2-Mini-SQ-10, all experts in VRAM, `--kv fp8`.

**Summary.** Enabling the int8/SF4 grouped-MoE path on an RDNA2 (`gfx103x`) build produces roughly 2x prefill
throughput and catastrophically wrong output. The existing call site is protected by
`use_i8_moe_wmma_ = (gen == AmdGen::Rdna4)`, so this is normally unreachable. PR #7009 introduces a second
entry point gated **only** on an env var and a batch size, with no architecture check, which would expose the
same broken data path on RDNA2 hardware.

**Reproducibility caveat, stated up front.** On a stock build these numbers are NOT reachable, because
`use_i8_moe_wmma_` is hard-wired to RDNA4 and `STRATA_HIP_MOE_I8` cannot override it on gfx1031. Getting the
table below required a local 22-line patch relaxing that assignment so the env var takes effect on `gfx103x`
(the kernels themselves compiled and ran unchanged). This is exactly why the #7009 note below matters: that PR
makes a comparable data path reachable with an env var alone, on any architecture, with no local patch needed.

**Measured, correct path vs int8 path** (patched build; `STRATA_HIP_MOE_I8` unset vs `=1`, same binary, same prompt):

| metric | correct path | int8/SF4 path |
|---|---:|---:|
| prefill ppts @ prompt 2048 | 187.0 | **381.7 / 381.8** (x2.05) |
| prefill ppts @ prompt 8192 | 158.7 | 282.9 |
| Mean KLD vs reference logits | 0.000000 | **3.590106** |
| Top-1 agreement | 100% | **46.18%** |
| Perplexity | 1,101.388 | **58,222.513** |
| decode tps | 51.1 | 44.0 |

Decode drops despite the faster prefill because verify logits are wrong, which collapses draft acceptance --
the slowdown is downstream of the numerics, not a separate defect.

**Ruled out** (i.e. this is not a routing bug on my side):
- wrong codec routing: the model's expert weights are `CODE_SF4`, which maps to `layer_gu_class_ == 1`, and the
  SF4 kernel (`k_moe_grouped_w13_situ_wmma_sf4_i8_128`) provably executed.
- missing activation hand-off between w13 and w2: `k_quant_f16_to_q8_16` is invoked.

The fault appears to be inside the int8 kernels' RDNA2 lowering. Note this path is inherently not bit-identical
to the bf16 path anyway (q8_16 quantized activations), so "fast but wrong" here is not a tolerance question --
PPL 58,222 vs 1,101 is a broken data path, not a rounding difference.

**The #7009 concern.** Existing int8 dispatch is arch-guarded:

```cpp
use_i8_moe_wmma_ = (gen == AmdGen::Rdna4)   // ink_hip_backend.hip.cpp:760
```

PR #7009's new gate is env + batch size only:

```cpp
bool topup_moe_on(int B) const { return topup_moe_ && B >= 64 && B <= topup_maxb_; }
// topup_moe_ = (getenv("STRATA_INK_TOPUP_MOE") != 0)
```

On a gfx1031 build, setting `STRATA_INK_TOPUP_MOE=1` selects `k_moe_grouped_{w13_situ,w2}_wmma_sf4_i8_n8`
with no matrix cores available, and per the table above yields silently wrong text rather than an error.
Suggest the new path inherit the same `use_i8_moe_wmma_` / arch guard, or refuse with a diagnostic on
non-RDNA4.

**Repro.**
```sh
LD_LIBRARY_PATH=... ./strata-inkling --native MODEL --parallel 1 --kv fp8 \
  --vram-experts-gb 8.77 --ram-experts-gb 0 \
  --prompt-ids @prompt512_cma.ids --kld-eval ref_512_c256.bin
# reference logits produced by the same build with the int8 env var unset, then compared with --kld-eval
# NOTE: the int8 side additionally needs the arch guard lifted locally (see Reproducibility caveat above)
```

**Relevance to #6850** (`feat/gfx103x-mmq-prefill`, "no card report yet"): that PR is explicitly aimed at
`gfx103x`. If it reuses the SF4 int8 MoE data path, it inherits this failure on RDNA2. Happy to test any
branch; the parity harness (`--dump-ref` / `--kld-eval`) makes the check mechanical -- correct builds score
`KLD=0.000000`, this scores 3.59.

---

## Issue B -- HIP fused speculative path has no acceptance-based throttle, so speculation stays fully on when it is net-negative

**Symptom.** Speculative decoding is net **negative** on short context and strongly positive elsewhere.
gen 256 (steady state), RX 6700 XT, Inkling-2-Mini-SQ-10:

| prompt | spec ON | `--draft off` | delta |
|---|---:|---:|---:|
| 512 | 25.8 tps | **34.2 tps** | **-24.6%** |
| 2048 | **51.0 tps** | 32.8 tps | +55.5% |
| 8192 | **41.9 tps** | 30.7 tps | +36.5% |

At prompt 512 the cost is a flat `+9.6 ms/token` over draft-off (38.8 vs 29.2 ms) on top of an otherwise
identical ~30 ms full forward -- i.e. drafts are being produced and paid for, and essentially nothing is
accepted. (I cannot yet say *which* verify shape this corresponds to: `STRATA_HIP_PROF=2` showed only `B=1` and
`B=512` buckets during a decode run, with no `B=2..8`, yet both candidate paths would be expected to produce
batched verify batches. Most likely graph replay hides those launches from the tracer. A `--spec-check` run prints
`steps=N (fused X, draft+verify Y, single Z), drafted D, emitted E` (`ink_main.cpp:743`), which settles both which
path executes and the acceptance rate; that run is the missing evidence and is why this section states the sign and
size of the effect as measured, and the mechanism only as "no throttle exists in the fused path", which is a
property of the code rather than an inference.)

**CORRECTED after checking which path the bench actually runs** -- an earlier version of this report blamed a
`keep_steps >= 1` guard in `rollback_slot()` for a "stuck EMA". That was wrong: that logic lives in
`dspark_draft()`, which `ink_main.cpp:686` only reaches as a *fallback*. The primary path is
`InkBackend::dspark_spec_step_greedy()`, and HIP implements it (`ink_hip_backend.hip.cpp:2174`).

**Actual mechanism.** The fused path drafts a **fixed-size tree** (`kTreeMaxM`, `B_ds = dspark_cfg_.block_size` = 8,
tree scratch pre-allocated via `ensure_tree_scratch`) and contains **no acceptance-based throttling at all** -- no
`adapt_max`, no `dspark_ema_acc`, no skip/backoff. It pays full draft+tree-verify cost every step regardless of how
much of the tree is being accepted. The EMA/backoff logic exists only on the fallback path, which this configuration
does not use. So when acceptance collapses, nothing turns speculation off, and it is a pure tax.

(Secondary, clearly labelled as the *fallback* path and therefore NOT implicated in the numbers above:
`dspark_draft()`'s decay is gated by `if (sl.dspark_last_draft > 0 && keep_steps >= 1)` in `rollback_slot()`, where
`keep_steps = keep_len - sl.snap_seq_len`. If total rejection returns to the snapshot length, `keep_steps == 0` and
the EMA update is skipped. Whether that state is reachable depends on snapshot bookkeeping I have not traced. Reported
only in case it matters to the fallback path; it does not explain this benchmark.)

Old (withdrawn) framing, kept for transparency -- the EMA is pushed up in `dspark_draft()` on full acceptance
(`:1958`) and decays only inside `rollback_slot()`, behind this guard:

```cpp
const int keep_steps = keep_len - sl.snap_seq_len;
if (sl.dspark_last_draft > 0 && keep_steps >= 1) {          // <-- total rejection gives keep_steps == 0
    const int acc_toks = std::max(0, std::min(sl.dspark_last_draft, keep_steps - 1));
    const float rate = float(acc_toks) / float(sl.dspark_last_draft);
    sl.dspark_ema_acc = 0.70f * sl.dspark_ema_acc + 0.30f * rate;
```

A step that rejects every draft returns to the snapshot length, so `keep_steps == 0` and the EMA update is
skipped entirely. Consequence: the worse acceptance gets, the less the controller learns; `dspark_ema_acc` stays
near its 0.50 seed, permanently above the 0.20 / 0.25 backoff thresholds, so `adapt_max` stays at `B-1` and the
full draft window is paid every step indefinitely. The code comment at `:1954` advertises a `K=4->2->0` backoff
that the implementation cannot reach, since the lowest reachable `adapt_max` is 2 and it is only entered when the
EMA already moved.

**Not a claim of new research.** "Low acceptance makes speculation pure overhead" and EWMA-controlled draft
length are well established; the point is narrow -- this particular decay guard makes the low-acceptance state
unobservable, so the adaptive logic that exists cannot engage.

**Question before proposing anything:** is the fixed 8-wide tree in `dspark_spec_step_greedy` intended to apply at
every acceptance level on HIP, or should the fused path honour an acceptance/realized-gain floor and fall back to
plain decode when speculation is losing (the way the fallback path tries to with `adapt_max`)? Also worth noting: changing draft counts is **not** bit-identical
(it changes which tokens are sampled for a fixed RNG stream) even though accept/reject sampling preserves the
output distribution -- anyone gating CI on exact output equality should know that up front.
