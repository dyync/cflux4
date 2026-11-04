# cflux4

FLUX.1-dev text-to-image with a **Gradio frontend** (port 7860) and a **GPU inference backend** (port 8000) exposing a vLLM/OpenAI-compatible images API. CUDA 12.9, Docker Compose.

```
 browser ──► frontend (Gradio, :7860) ──HTTP──► backend (FastAPI + diffusers FluxPipeline, CUDA 12.9, :8000)
```

> **Why not plain `vllm serve`?** vLLM's main server is built for LLMs and has no FLUX diffusion pipeline. The backend runs the exact `FluxPipeline` code (bf16, 1024², guidance 3.5, 50 steps, max_seq 512) behind the same style of OpenAI-compatible HTTP API (`/v1/models`, `/v1/images/generations`, `/health`), so any OpenAI-images client works against it.

## Requirements

- Linux host (or Windows + WSL2) with an NVIDIA GPU, driver **≥ 575** (CUDA 12.9)
- Docker + Compose v2 + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- A Hugging Face token that has accepted the [FLUX.1-dev license](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- ~35 GB disk for weights (cached in the `hf-cache` volume)

## Quick start

```bash
git clone https://github.com/dyync/cflux4.git && cd cflux4
cp .env.example .env        # put your HF_TOKEN in .env
docker compose up -d --build
docker compose logs -f backend   # first start downloads ~33 GB of weights
```

Open **http://localhost:7860**. The status line shows `loading` until the model is ready, then `ready`.

## VRAM / offload modes (`OFFLOAD_MODE`)

| Mode | What it does | GPU |
|---|---|---|
| `auto` (default) | picks below by detected VRAM | any |
| `gpu` | `pipe.to("cuda")` | ≥ 36 GB (A100, H100, L40S, RTX 6000) |
| `model` | `pipe.enable_model_cpu_offload()` | 16–24 GB (RTX 3090/4090/5080) |
| `sequential` | `pipe.enable_sequential_cpu_offload()` | < 12 GB (slow) |

## API

```bash
curl http://localhost:8000/health
curl -s http://localhost:8000/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"A cinematic shot of a futuristic sports car in a neon cyberpunk city at night",
       "size":"1024x1024","num_inference_steps":50,"guidance_scale":3.5,
       "max_sequence_length":512,"seed":42}' \
  | python3 -c "import sys,json,base64;open('out.png','wb').write(base64.b64decode(json.load(sys.stdin)['data'][0]['b64_json']))"
```

Request fields: `prompt`, `size` (`WxH`, multiples of 16), `n` (1–4), `num_inference_steps`, `guidance_scale`, `max_sequence_length`, `seed` (`-1` = random). Response: `{"data":[{"b64_json":...}], "seed":..., "elapsed_seconds":...}`.

## Troubleshooting

- `HF_TOKEN` missing → compose refuses to start; set it in `.env`.
- `401/403` in backend logs → accept the model license on Hugging Face with the same account as the token.
- `CUDA out of memory` → set `OFFLOAD_MODE=model` (or `sequential`) and `docker compose up -d`.
- `could not select device driver "nvidia"` → install NVIDIA Container Toolkit and restart Docker.
