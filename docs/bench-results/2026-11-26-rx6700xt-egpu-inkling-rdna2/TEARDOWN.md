# Teardown — returning this laptop to ordinary use

Written while the rig was still live, so the paths and commands are accurate. This box is the
Dell Latitude 5420 that was running the RX 6700 XT eGPU experiments; it is destined to go back to
being an ordinary laptop.

## Read this first: most of the setup is already ephemeral

`/` is an overlay over live squashfs (`/cow`), so the **entire Linux side lives in RAM**. A reboot
discards, with no action needed:

- the `strata-inkling` systemd unit and its `enabled` symlink (I enabled it, but it does not persist)
- the installed Strata Terminal CLI and `~/.local/share/strata` (engine components, ROCm libraries, run profiles)
- `/etc/strata/inkling.env` (the serving API key)
- `kernel.perf_event_paranoid=2` — raised from 4 for hardware-counter profiling, **runtime only**, no
  `/etc/sysctl.conf` or `/etc/sysctl.d` rule was written, so a reboot restores 4

So "hand the laptop back" is mostly: reboot, then deal with the Windows volume.

## What actually persists and must be handled

1. **The GitHub credential, in plaintext on the Windows drive** — `/mnt/win/llm-temp/.gh/auth.log`.
   This is the item most likely to be forgotten, and the Windows volume *keeps being used*. Revoke or
   rotate that token at https://github.com/settings/tokens before or instead of deleting the file.
2. **The model and build artefacts, ~15 GB on the Windows drive** — `rm -rf /mnt/win/llm-temp`.
   Contains the 11 GB model set, 2.3 GB parity logits, 274 MB engine package, 76 MB build tree.
   Nothing in it is unique: the source, tooling, report and deploy templates are all on
   `myfork/rdna2-moe-lds-staging`.
3. **Port 22 open on all interfaces** (`0.0.0.0:22`) — pre-existing, not from this work, but it is the
   one thing that outlives a reboot and matters on a non-technical user's laptop. Note the surprise:
   `systemctl is-enabled ssh` reports `disabled` and `sshd` reports `alias`, yet 22 is listening, so
   find what is actually serving it before assuming it is off.
4. **The eGPU itself** — unplug the Razer Core at Thunderbolt. Nothing in BIOS/boot needs undoing.

## If the service is still running when you tear down

```sh
sudo systemctl stop strata-inkling          # stops 0.0.0.0:8095
sudo systemctl disable strata-inkling
sudo rm -f /etc/systemd/system/strata-inkling.service && sudo systemctl daemon-reload
sudo rm -f /etc/strata/inkling.env
ss -ltn | grep -E ':8095|:8080' || echo "no Strata listeners remain"
```

The unit binds `0.0.0.0:8095` with an API key. With `STRATA_API_KEY` set, the DNS-rebinding Host
check (#7130) is skipped by design, so that listener is reachable from anything on the LAN for as
long as it runs. Stop it before the laptop joins a network it isn't meant to serve.

## Nothing here is lost

`myfork/rdna2-moe-lds-staging` carries the whole effort: 8 code commits. Five of them are bit-identical
and were gated on `--kld-eval Mean KLD=0.000000 | Top-1=100.00%` against the committed reference logits —
`3974192` LDS staging, `d613831`/`9df38f5` the two occupancy bumps, `a0e388d` the chunk cap, `2460e04a3` the
decode bump. The remaining three (`1811377`, `b2de3d4`, `965fabf`) are the initial RDNA2 dense-GEMM/attention
fast paths and the chunk-override plumbing; `1811377` (RDNA2 dense-GEMM + attention split-KV) is **not** bit-identical at the
logit level: the working log records that greedy token sequences matched between builds (`371, 150409,
150008`) while **logit hashes differed slightly** from float reassociation in the split-KV merge. That is
expected for a reassociation and was accepted as such, but it means the reference logits are only comparable
from `3974192` forward, not across `1811377`. `b2de3d4`/`965fabf` are the chunk-override plumbing, whose
default path was KLD-verified as unchanged. Plus this
`docs/bench-results/2026-11-26-rx6700xt-egpu-inkling-rdna2/`
folder with the curated report, the raw 31-section working log, the two unposted upstream issue
drafts, the measurement scripts and micro-benchmarks, and `deploy/restore.sh` which rebuilds the
serving deployment on any RDNA2 box. Re-cloning the branch and re-fetching the model is enough to
resume the work on different hardware.

## Unresolved when this was written

- Serving was never started, so the served configuration (`--draft off`, `--ram-experts-gb 1.69`,
  128 K context) has **no measured decode throughput**. The benchmark numbers in `README.md` are
  `--bench` with all experts in VRAM and speculation on; they are not what the service would deliver.
- The concurrency sweep was stopped during `--parallel 4` once `--parallel 3` showed aggregate throughput
  collapsing (see the concurrency table in `README.md`). It did **not** yield a win: on this card concurrency
  is net-negative, so that line of attack is closed rather than unfinished.
- `--draft off` in the served profile costs the +55% speculation gain measured on this card.
