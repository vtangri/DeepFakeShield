#!/usr/bin/env bash
# DeepFakeShield — Kaggle GPU training control script.
#
#   ./scripts/kaggle_train.sh push     Upload ml/kaggle/kernel.ipynb and start a GPU run
#   ./scripts/kaggle_train.sh status   Print current kernel status
#   ./scripts/kaggle_train.sh wait     Poll until the run completes or errors
#   ./scripts/kaggle_train.sh pull     Download trained weights + evaluation into the repo
#   ./scripts/kaggle_train.sh run      push -> wait -> pull
#
# Requires a configured Kaggle CLI (see scripts/setup_kaggle.py).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KERNEL_DIR="$REPO_ROOT/ml/kaggle"
KERNEL_ID="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['id'])" "$KERNEL_DIR/kernel-metadata.json")"
OUTPUT_DIR="$REPO_ROOT/ml/kaggle/output"
MODELS_DIR="$REPO_ROOT/ml/models"
EVAL_DIR="$REPO_ROOT/ml/evaluation"
POLL_SECONDS="${POLL_SECONDS:-60}"

command -v kaggle >/dev/null || { echo "[ERROR] kaggle CLI not found. pip install kaggle"; exit 1; }

kernel_status() {
  # Prints one of: RUNNING | COMPLETE | ERROR | CANCEL_REQUESTED | QUEUED ...
  kaggle kernels status "$KERNEL_ID" 2>&1 | sed -n 's/.*KernelWorkerStatus\.\([A-Z_]*\).*/\1/p'
}

case "${1:-run}" in
  push)
    echo "[INFO] Pushing $KERNEL_ID (GPU enabled)..."
    kaggle kernels push -p "$KERNEL_DIR"
    ;;

  status)
    kaggle kernels status "$KERNEL_ID"
    ;;

  wait)
    echo "[INFO] Waiting for $KERNEL_ID (polling every ${POLL_SECONDS}s)..."
    while true; do
      s="$(kernel_status)"
      echo "[$(date +%H:%M:%S)] status=${s:-UNKNOWN}"
      case "$s" in
        COMPLETE) echo "[SUCCESS] Run finished."; break ;;
        ERROR|CANCEL_ACKNOWLEDGED) echo "[ERROR] Run ended with status $s."; exit 1 ;;
      esac
      sleep "$POLL_SECONDS"
    done
    ;;

  pull)
    echo "[INFO] Downloading outputs to $OUTPUT_DIR ..."
    mkdir -p "$OUTPUT_DIR" "$MODELS_DIR" "$EVAL_DIR"
    kaggle kernels output "$KERNEL_ID" -p "$OUTPUT_DIR"

    # Kaggle flattens /kaggle/working, so weights may land at the top level
    # or under models/ depending on how the notebook wrote them.
    found=0
    while IFS= read -r f; do
      cp "$f" "$MODELS_DIR/"
      echo "  weights -> ml/models/$(basename "$f")"
      found=1
    done < <(find "$OUTPUT_DIR" -name '*.pt' -type f)
    [ "$found" -eq 1 ] || echo "[WARNING] No .pt weights found in kernel output."

    while IFS= read -r f; do
      cp "$f" "$EVAL_DIR/"
      echo "  eval    -> ml/evaluation/$(basename "$f")"
    done < <(find "$OUTPUT_DIR" \( -name '*_metrics.json' -o -name '*_report.txt' -o -name '*_roc.png' -o -name '*_cm.png' \) -type f)

    echo
    echo "[INFO] Backend expects (see backend/app/core/config.py):"
    echo "  ml/models/video_forensics_final.pt"
    echo "  ml/models/audio_spoof_final.pt"
    ls -la "$MODELS_DIR"
    ;;

  run)
    "$0" push && "$0" wait && "$0" pull
    ;;

  *)
    echo "Usage: $0 {push|status|wait|pull|run}"; exit 1 ;;
esac
