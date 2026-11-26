#!/bin/bash
# Recreate the served-instance deployment after a reboot or a /mnt/win cleanup.
#
# WHY THIS EXISTS: on this box `/` is an overlay over live squashfs (`/cow`), so the whole
# Linux side -- /home/ubuntu, /etc/systemd/system, ~/.local/share/strata -- lives in RAM and
# does NOT survive a reboot. Only /mnt/win persists. Anything this script needs from the RAM
# side has to be re-created, which is why it is here instead of in /etc.
#
# It deliberately does not try to be clever: it checks each prerequisite and tells you the
# exact next command instead of failing deep inside systemd.
set -eu

SRC=${SRC:-/mnt/win/llm-temp/strata-src}
DOC="$SRC/docs/bench-results/2026-11-26-rx6700xt-egpu-inkling-rdna2"
ENGINE_DIR=${ENGINE_DIR:-/mnt/win/llm-temp/engines/inkling2-rdna2-local}
MODEL_DIR=${MODEL_DIR:-/mnt/win/llm-temp/strata-models}
PROFILE_DST=/home/ubuntu/.local/share/strata/install/strata-inkling-2-mini-sq-10-rdna2.json
UNIT=/etc/systemd/system/strata-inkling.service
ENVF=/etc/strata/inkling.env

say() { printf '%s\n' "$*"; }
need() { say "  MISSING: $1"; say "         -> $2"; MISSING=1; }

MISSING=0
say "== prerequisites =="

[ -x /home/ubuntu/.local/bin/strata ] \
  || need "Strata Terminal CLI" "reinstall Strata Terminal, then run: strata update   (it lives in RAM, so a reboot removes it)"
[ -d /home/ubuntu/.local/share/strata/components/shapes/inkling2/engine ] \
  || need "stock hip-rx6000 engine component" "strata update   (also RAM-resident; the ROCm libs our build links against come from it)"
[ -x "$ENGINE_DIR/strata-inkling" ] \
  || need "local engine build" "rebuild then reassemble:
           cd $SRC && cmake --build /mnt/win/llm-temp/build-ink --target strata-inkling -j\"\$(nproc)\"
           mkdir -p $ENGINE_DIR
           cp /mnt/win/llm-temp/build-ink/src/inkling2/strata-inkling $ENGINE_DIR/
           cp -aL /mnt/win/llm-temp/enginetest/lib*.so* $ENGINE_DIR/
           cp $DOC/deploy/BUILD.json.local $ENGINE_DIR/BUILD.json 2>/dev/null || true"
[ -f "$MODEL_DIR/models/inkling-2-mini/Inkling-2-Mini-SQ-10.strata" ] \
  || need "model file" "restore $MODEL_DIR from wherever the 11 GB model set was fetched from (verify you CAN re-fetch before deleting it)"
[ -f "$PROFILE_DST" ] || [ -f "$DOC/deploy/profile.local-engine.json" ] || need "profile template" "the template is committed in this branch; re-clone $SRC"

[ "$MISSING" = 0 ] || { say ""; say "Resolve the above, then re-run this script."; exit 1; }

say ""; say "== installing profile, key, unit =="
mkdir -p "$(dirname "$PROFILE_DST")" /etc/strata
install -m 0644 "$DOC/deploy/profile.local-engine.json" "$PROFILE_DST"
say "  profile -> $PROFILE_DST"

if [ ! -f "$ENVF" ]; then
  printf 'STRATA_API_KEY=%s\n' "$(openssl rand -hex 24)" | sudo tee "$ENVF" >/dev/null
  sudo chmod 600 "$ENVF"
  say "  new API key written to $ENVF (0600). Read it with: sudo cat $ENVF"
  say "  NOTE: with STRATA_API_KEY set, the #7130 DNS-rebinding Host check is skipped, which is what"
  say "        reaching the server on 0.0.0.0 by LAN IP needs. Without a key, remote requests get 403."
else
  say "  keeping existing $ENVF"
fi

sudo tee "$UNIT" >/dev/null < "$DOC/deploy/strata-inkling.service"
sudo systemctl daemon-reload
sudo systemctl enable strata-inkling >/dev/null 2>&1
say "  unit installed and enabled at boot: $UNIT"

say ""
say "Start it (loads ~26B into the 12 GiB eGPU -- make sure nothing else is on the GPU):"
say "  sudo systemctl start strata-inkling && journalctl -u strata-inkling -f"
say "Then verify:"
say "  curl -s -H 'Authorization: Bearer <key>' http://127.0.0.1:8095/v1/models | head -c 400"
