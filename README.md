# Typed decisions from scratch

Code for the blog series [*Typed decisions from scratch*](https://liminma.hashnode.dev/series/typed-decisions-from-scratch) on **typed decisions**, which shows how to get a local LLM to return a label from a predefined set with calibrated confidence instead of free text.

Each post covers one idea. Each folder here holds the runnable code for one post. Temperature scaling, ECE and the threshold search are all written out in a few dozen lines.

## What you need

- **A GB10 machine:** The series was developed and timed on a **Dell Pro Max with GB10**. Other NVIDIA GPUs with enough memory for the model should also work.
 But you'll need to adapt the setup in [`00-setup/build-server.sh`](00-setup/build-server.sh) for your GPU and driver.
- **Docker** with the NVIDIA container toolkit.
- **[uv](https://docs.astral.sh/uv/)** for the Python side.

The model is [Ternary Bonsai 2 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf), a GGUF file of about 7 GB, served by `llama-server` from the [PrismML llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp). Stock llama.cpp refuses this file. [`00-setup/`](00-setup/) builds the server image, downloads the model and starts the server.

The image is built from source on your machine, with the fork's own Dockerfile at the commit the series used, compiled for GB10. The series' own code only makes plain HTTP calls to `llama-server`, so it can also run on a different machine from the server.

## Layout

| Folder | Post |
|---|---|
| `00-setup/` | Build the server image, download the model, start `llama-server` |
| `datasets/` | BANKING77 and CLINC150: label groups, descriptions, fixed splits; how to make your own |
| `01-why-typed-decisions/` | Why typed decisions? |
| `02-forcing-a-valid-answer/` | Forcing a valid answer: grammars and single-token labels |
| `03-where-confidence-comes-from/` | Where confidence comes from: logprobs |
| `04-is-the-confidence-honest/` | Is the confidence honest? Reliability plots and ECE |
| `05-temperature-scaling/` | Fixing it: temperature scaling |
| `06-isotonic-regression/` | When one knob isn't enough: isotonic regression |
| `07-accuracy-coverage-and-auroc/` | Does confidence separate right from wrong? |
| `08-from-confidence-to-action/` | From confidence to action: thresholds and routing |

Folders appear as their posts are published. Each post links to a tag (`post-01`, `post-02`, …), so the code a post shows doesn't change under it.

## Running

```
uv sync
```

Then follow `00-setup/` to start the server, and each folder's README for its scripts. The datasets' messages aren't stored in this repo; scripts download them on first use. [`datasets/`](datasets/) holds what the series adds: label groups, descriptions and fixed splits.

## License

The code is [Apache-2.0](LICENSE). The datasets and the model keep their own licences: BANKING77 is CC BY 4.0, CLINC150 is CC BY 3.0 (credits in [`datasets/`](datasets/)), and Ternary Bonsai 2 27B is Apache-2.0.
