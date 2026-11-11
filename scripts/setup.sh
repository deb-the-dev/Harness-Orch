#!/usr/bin/env bash
# Create the venv, install dependencies, and fetch OPA with checksum verification. Linux and macOS, x86_64 and arm64.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python3}
"$PY" - <<'PYEOF' || { echo "Python 3.12 or newer is required (set PYTHON=/path/to/python3.12)"; exit 1; }
import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)
PYEOF

"$PY" -m venv .venv
.venv/bin/pip install -q -e ".[test]"

case "$(uname -s)-$(uname -m)" in
  Linux-x86_64)            asset=opa_linux_amd64_static ;;
  Linux-aarch64|Linux-arm64) asset=opa_linux_arm64_static ;;
  Darwin-arm64)            asset=opa_darwin_arm64_static ;;
  Darwin-x86_64)           asset=opa_darwin_amd64 ;;
  *) echo "unsupported platform $(uname -s)-$(uname -m): download OPA by hand into .tools/opa/opa"; exit 1 ;;
esac

mkdir -p .tools/opa
if [ ! -x .tools/opa/opa ]; then
  base=https://github.com/open-policy-agent/opa/releases/latest/download
  curl -sSL -o .tools/opa/opa "$base/$asset"
  curl -sSL -o .tools/opa/opa.sha256 "$base/$asset.sha256"
  want=$(cut -d' ' -f1 .tools/opa/opa.sha256)
  if command -v sha256sum >/dev/null; then got=$(sha256sum .tools/opa/opa | cut -d' ' -f1); else got=$(shasum -a 256 .tools/opa/opa | cut -d' ' -f1); fi
  [ "$want" = "$got" ] || { echo "OPA checksum mismatch"; rm -f .tools/opa/opa; exit 1; }
  chmod +x .tools/opa/opa
fi
.tools/opa/opa version | head -1
echo "ok. Next: scripts/pg_dev.sh (needs Docker) then .venv/bin/python -m pytest -q"
