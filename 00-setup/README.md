# 00 - Setup: the llama.cpp server

The local server `llama-server` runs Ternary Bonsai 2 27B in a Docker container on `http://127.0.0.1:8080` by default. It is built in three steps on GB10-based hardware, such as the NVIDIA DGX Spark or Dell Pro Max with GB10. Other NVIDIA GPUs should also work, but you'll need to adapt the build (see [Other GPUs](#other-gpus) below).

| Script | Does | Takes |
|---|---|---|
| `build-server.sh` | Builds the server docker image from the PrismML llama.cpp fork | ~3.3 GB image (on GB10) |
| `download-model.sh` | Downloads the model and its vision encoder, and checks their SHA-256 sums | 7.3 GB on disk |
| `start-server.sh` | Starts the server and waits until the model is ready |  |

## Before you start

- **Docker with the NVIDIA Container Toolkit.** `docker info | grep -i runtimes` should list `nvidia`.
- **An NVIDIA driver.** run command `nvidia-smi` to check.
- **[uv](https://docs.astral.sh/uv/)**. `download-model.sh` uses `uv` to run the Hugging Face CLI.

## 1. Build the server image

```bash
./00-setup/build-server.sh
```

This clones the PrismML fork at tag [`prism-b10735-842b188`](https://github.com/PrismML-Eng/llama.cpp/tree/842b1880415d6f508f03b789e5ce70194def7bfd). It checks the tag still points at commit `842b188…`, and then builds the image `prism-llamacpp:b10735` using the fork's `.devops/cuda.Dockerfile`. 

The script passes two build arguments for GB10 platform:

- `CUDA_VERSION=13.0.1`. The Dockerfile's default CUDA version is 12.8.1. GB10 (`sm_121`) needs CUDA 12.9 or later.
- `CUDA_DOCKER_ARCH=121a-real`. Compiles for GB10 only. This makes the build shorter and the image smaller.

At the end, the script checks the build log for the line CMake prints (`Using CMAKE_CUDA_ARCHITECTURES=121a-real`). The full log is in `.build/`. Once the server is running, `curl -s 127.0.0.1:8080/props` should report `"build_info": "b10735-842b1880"`.

### Other GPUs

The series has only been run on GB10. On a different NVIDIA GPU, override both build arguments when you run the script.

## 2. Download the model

```bash
./00-setup/download-model.sh
```

This downloads two files from [prism-ml/Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) at the revision `8b71575…`, then checks them against [`SHA256SUMS`](SHA256SUMS):

| File | Size | Used for |
|---|---:|---|
| `Ternary-Bonsai-2-27B-PQ2_0.gguf` | 6.8 GB | The model |
| `Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf` | 0.59 GB | The vision encoder |

Set `MODEL_DIR` to download somewhere else, and pass the same `MODEL_DIR` to `start-server.sh`.
The models are saved in the `models/` folder by default.

## 3. Start the server

```bash
./00-setup/start-server.sh
```

It starts a container named `bonsai`. Set the following variables to change settings:

| Variable | Default | |
|---|---|---|
| `SLOTS` | `8` | Parallel requests |
| `CTX` | `131072` | Total context (shared evenly between the slots) |
| `VISION` | `0` | `1` loads the vision encoder |
| `PORT` | `8080` | Host port |
| `MODEL_DIR` | `models/` | Where `download-model.sh` put the files |
| `IMAGE`, `NAME` | `prism-llamacpp:b10735`, `bonsai` | Image and container name |

## 4. Check it

```bash
curl -s 127.0.0.1:8080/health
curl -s 127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "Is the sky blue? Answer yes or no."}],
  "temperature": 0, "max_tokens": 1}'
```

The answer should be a short `Yes` with no reasoning before it. Command `nvidia-smi` should list `/app/llama-server`, which means the model is on the GPU.

## Other GPUs

The series covers GB10 only, but the fork's Dockerfile can also be used for other NVIDIA GPUs by changing `CUDA_VERSION` and `CUDA_DOCKER_ARCH` (both can be set when running `build-server.sh`).

## Troubleshooting

- **`CUDA driver version is insufficient for CUDA runtime version`:** the image's CUDA is newer than your driver supports. Rebuild with a lower `CUDA_VERSION`.
- **Slow first start, or slow generation:** the image was probably compiled for another GPU. Check the `Using CMAKE_CUDA_ARCHITECTURES=` line in `.build/build-*.log`.
- **The container stops while loading:** usually out of GPU memory. Lower `CTX` or `SLOTS`. `docker logs bonsai` shows the error.
- **`image input is not supported`:** the server was started without `VISION=1`.
- **`error: .build/llama.cpp is not a full checkout of …`:** that folder holds another checkout of the fork, or a shallow one. Delete it and build again.
