#!/usr/bin/env bash
# Run scripts/ci/run_suite.py on another machine against one wheel: copy the wheel, its .sha256 and the test kit
# over ssh, run the suite there with the suite script taken from the test kit itself, and fetch result.json back.
#
# Usage:
#   scripts/ci/remote_suite.sh --ssh "ssh Server-5090" --remote-dir /path/on/remote/run1 \
#       --wheel dist/fish_scales_ops-0.2.0-…whl --testkit dist/fish_scales_ops-0.2.0-testkit-<commit>.tar.gz \
#       [--out DIR] -- --python /venv/bin/python --gpu-uuid GPU-… [--gpu-lock FILE] [--compute-mode default] \
#                      [--pythonpath-extra DIR] [--layer-ref REF.pt]
#
# Everything after `--` is passed to run_suite.py on the remote machine (see its docstring); --wheel, --testkit and
# --work are filled in by this script. `--ssh` is the complete ssh command for the machine, options included. Files
# are sent through `ssh … 'cat > file'`, so no scp configuration is needed. The remote directory is created; an
# existing file of the same name is overwritten only when its sha256 differs.
#
# Exit code: that of the remote suite. `--out DIR` (default: the wheel's directory) receives
# <host-tag>.result.json and <host-tag>.suite.log, where <host-tag> is the basename of --remote-dir.
set -euo pipefail

SSH="" REMOTE_DIR="" WHEEL="" KIT="" OUT=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ssh) SSH="$2"; shift 2;;
        --remote-dir) REMOTE_DIR="$2"; shift 2;;
        --wheel) WHEEL="$2"; shift 2;;
        --testkit) KIT="$2"; shift 2;;
        --out) OUT="$2"; shift 2;;
        --) shift; break;;
        *) echo "remote_suite.sh: unknown argument $1" >&2; exit 2;;
    esac
done
[[ -n "$SSH" && -n "$REMOTE_DIR" && -f "$WHEEL" && -f "$KIT" ]] || {
    echo "usage: remote_suite.sh --ssh CMD --remote-dir DIR --wheel W --testkit K [--out DIR] -- <run_suite.py args>" >&2
    exit 2
}
OUT="${OUT:-$(dirname "$WHEEL")}"
TAG="$(basename "$REMOTE_DIR")"
mkdir -p "$OUT"

send() {  # send <local file> <remote name>: skip the copy when the remote file already has the same sha256
    local src="$1" dst="$REMOTE_DIR/$2" sum
    sum="$(sha256sum "$src" | cut -d' ' -f1)"
    if [[ "$($SSH "sha256sum '$dst' 2>/dev/null | cut -d' ' -f1" || true)" == "$sum" ]]; then
        echo "[remote] $2 already there (sha256 ${sum:0:16})"
    else
        $SSH "cat > '$dst'" < "$src"
        echo "[remote] sent $2 (sha256 ${sum:0:16})"
    fi
}

$SSH "mkdir -p '$REMOTE_DIR'"
send "$WHEEL" "$(basename "$WHEEL")"
[[ -f "$WHEEL.sha256" ]] && send "$WHEEL.sha256" "$(basename "$WHEEL").sha256"
send "$KIT" "$(basename "$KIT")"
# The suite script comes from the test kit, so the remote run uses the tests and the runner of the wheel's commit.
tar -xzOf "$KIT" --wildcards '*scripts/ci/run_suite.py' | $SSH "cat > '$REMOTE_DIR/run_suite.py'"

ARGS=""
for a in "$@"; do ARGS+=" $(printf '%q' "$a")"; done
PY=python3
for ((i = 1; i <= $#; i++)); do
    if [[ "${!i}" == "--python" ]]; then j=$((i + 1)); PY="${!j}"; fi
done

set +e
$SSH "cd '$REMOTE_DIR' && '$PY' run_suite.py --wheel '$REMOTE_DIR/$(basename "$WHEEL")' --testkit '$REMOTE_DIR/$(basename "$KIT")' --work '$REMOTE_DIR/work' $ARGS" 2>&1 | tee "$OUT/$TAG.suite.log"
RC=${PIPESTATUS[0]}
set -e
$SSH "cat '$REMOTE_DIR/work/result.json'" > "$OUT/$TAG.result.json" 2>/dev/null || echo "[remote] no result.json to fetch"
echo "[remote] suite exit $RC; log $OUT/$TAG.suite.log, result $OUT/$TAG.result.json"
exit $RC
