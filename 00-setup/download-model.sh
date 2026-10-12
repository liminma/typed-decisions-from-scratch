#!/usr/bin/env bash
# Download Ternary Bonsai 2 27B (PQ2_0, 7.2 GB) and its vision encoder (0.6 GB) from
# Hugging Face and then check their sha256sum.
#
# Usage:  ./download-model.sh
#         MODEL_DIR=/data/models ./download-model.sh
#
set -euo pipefail

REPO=prism-ml/Ternary-Bonsai-2-27B-gguf
REVISION=8b7157531df3859ce2a415c60754e9152554e25d

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
MODEL_DIR=${MODEL_DIR:-$ROOT/models}

uvx --from huggingface_hub hf download "$REPO" \
  Ternary-Bonsai-2-27B-PQ2_0.gguf \
  Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf \
  --revision "$REVISION" \
  --local-dir "$MODEL_DIR"

echo "Checking SHA-256 sums ..."
(cd "$MODEL_DIR" && sha256sum -c "$HERE/SHA256SUMS")
echo "Model files are in $MODEL_DIR"
