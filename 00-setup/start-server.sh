#!/usr/bin/env bash
# Start llama-server with Ternary Bonsai 2 27B:
#   - thinking off (--reasoning off): typed decision request wants a short answer, often one token
#       * configured with the model's sampling defaults of instruct mode
#   - CTX (default 131072), SLOTS (default 8): 16K per slot can hold a 150-label prompt with
#     examples (about 11K tokens)
#   - --cache-ram 0: disable prompt cache in host RAM. copying an 11K-token prompt to RAM adding
#     1.3-1.6 s to every request on GB10
#   - /metrics on to report latency
#   - optional VISION=1 to enable image input (default VISION=0)
#
# Usage:  ./start-server.sh
#         VISION=1 ./start-server.sh                # enable image input
# Stop:   docker rm -f bonsai

set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODEL_DIR=${MODEL_DIR:-$ROOT/models}
IMAGE=${IMAGE:-prism-llamacpp:b10735}
NAME=${NAME:-bonsai}
PORT=${PORT:-8080}
SLOTS=${SLOTS:-8}
CTX=${CTX:-131072}
VISION=${VISION:-0}

mmproj=()
if [ "$VISION" = 1 ]; then
  mmproj=(--mmproj /models/Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf)
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run -d --gpus all -p "127.0.0.1:$PORT:8080" \
  --name "$NAME" \
  -v "$MODEL_DIR":/models:ro \
  "$IMAGE" \
  -m /models/Ternary-Bonsai-2-27B-PQ2_0.gguf \
  "${mmproj[@]}" \
  --alias bonsai-2-27b \
  -ngl 99 \
  -c "$CTX" -np "$SLOTS" \
  --jinja \
  --reasoning off \
  --temp 0.7 --top-p 0.8 --top-k 20 --min-p 0.0 \
  --presence-penalty 1.5 --repeat-penalty 1.0 \
  --metrics \
  --cache-ram 0 >/dev/null

# Wait for the model to load
printf 'Loading the model'
for _ in $(seq 120); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then
    echo
    echo "Ready: http://127.0.0.1:$PORT (container $NAME)"
    exit 0
  fi
  if [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" != true ]; then
    echo
    echo "error: the container stopped. Its last log lines:" >&2
    docker logs --tail 30 "$NAME" >&2
    exit 1
  fi
  printf '.'
  sleep 2
done
echo
echo "error: not ready after 4 minutes. See: docker logs $NAME" >&2
exit 1
