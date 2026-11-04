"""
cflux4 backend - FLUX.1-dev inference server (port 8000).

Exposes a vLLM / OpenAI-compatible image API:
  GET  /health                    -> 200 when the model is loaded, 503 while loading
  GET  /v1/models                 -> model list (OpenAI format)
  POST /v1/images/generations     -> OpenAI images API + FLUX extras
"""
import asyncio
import base64
import io
import logging
import os
import random
import threading
import time
from contextlib import asynccontextmanager
from typing import Optional

import torch
from diffusers import FluxPipeline
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("cflux4.backend")

MODEL_ID = os.getenv("MODEL_ID", "black-forest-labs/FLUX.1-dev")
# auto | gpu | model | sequential
OFFLOAD_MODE = os.getenv("OFFLOAD_MODE", "auto").lower()
# Full bf16 FLUX.1-dev (transformer + T5 + CLIP + VAE) needs ~33 GB VRAM.
FULL_GPU_MIN_VRAM_GB = float(os.getenv("FULL_GPU_MIN_VRAM_GB", "36"))
MAX_PIXELS = int(os.getenv("MAX_PIXELS", str(2048 * 2048)))

state = {"pipe": None, "status": "starting", "error": None, "mode": None}
gpu_lock = threading.Lock()


def _choose_mode() -> str:
    if OFFLOAD_MODE in ("gpu", "model", "sequential"):
        return OFFLOAD_MODE
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available inside the container (check NVIDIA Container Toolkit).")
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    log.info("Detected GPU: %s with %.1f GB VRAM", torch.cuda.get_device_name(0), vram_gb)
    if vram_gb >= FULL_GPU_MIN_VRAM_GB:
        return "gpu"
    if vram_gb >= 12:
        return "model"
    return "sequential"


def load_pipeline() -> None:
    try:
        state["status"] = "loading"
        t0 = time.time()
        log.info("Loading %s (bfloat16)...", MODEL_ID)
        pipe = FluxPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            token=os.getenv("HF_TOKEN") or None,
        )
        mode = _choose_mode()
        if mode == "gpu":
            pipe.to("cuda")
        elif mode == "model":
            pipe.enable_model_cpu_offload()
        else:
            pipe.enable_sequential_cpu_offload()
        pipe.set_progress_bar_config(disable=True)
        state.update(pipe=pipe, mode=mode, status="ready")
        log.info("Model ready in %.1fs (mode=%s)", time.time() - t0, mode)
    except Exception as e:  # keep the server up so /health can report the error
        log.exception("Failed to load model")
        state.update(status="error", error=f"{type(e).__name__}: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=load_pipeline, daemon=True).start()
    yield


app = FastAPI(title="cflux4 backend", version="1.0.0", lifespan=lifespan)


class ImageRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    model: Optional[str] = None
    n: int = Field(1, ge=1, le=4)
    size: str = "1024x1024"
    response_format: str = "b64_json"
    # FLUX extras
    guidance_scale: float = Field(3.5, ge=0.0, le=20.0)
    num_inference_steps: int = Field(50, ge=1, le=100)
    max_sequence_length: int = Field(512, ge=64, le=512)
    seed: int = -1


def _parse_size(size: str) -> tuple[int, int]:
    try:
        w, h = (int(x) for x in size.lower().split("x"))
    except Exception:
        raise HTTPException(400, f"Invalid size '{size}', expected WIDTHxHEIGHT")
    if w % 16 or h % 16 or w < 256 or h < 256 or w * h > MAX_PIXELS:
        raise HTTPException(400, "Width/height must be multiples of 16, >=256, and within MAX_PIXELS")
    return w, h


def _generate(req: ImageRequest, width: int, height: int, seed: int):
    pipe = state["pipe"]
    with gpu_lock, torch.inference_mode():
        gen_device = "cuda" if state["mode"] == "gpu" else "cpu"
        generator = torch.Generator(gen_device).manual_seed(seed)
        images = pipe(
            req.prompt,
            height=height,
            width=width,
            guidance_scale=req.guidance_scale,
            num_inference_steps=req.num_inference_steps,
            max_sequence_length=req.max_sequence_length,
            num_images_per_prompt=req.n,
            generator=generator,
        ).images
    return images


@app.get("/health")
def health():
    body = {"status": state["status"], "model": MODEL_ID, "mode": state["mode"], "error": state["error"]}
    return JSONResponse(body, status_code=200 if state["status"] == "ready" else 503)


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "black-forest-labs"}]}


@app.post("/v1/images/generations")
async def generate(req: ImageRequest):
    if state["status"] != "ready":
        raise HTTPException(503, f"Model not ready (status={state['status']}, error={state['error']})")
    if req.response_format != "b64_json":
        raise HTTPException(400, "Only response_format='b64_json' is supported")
    width, height = _parse_size(req.size)
    seed = req.seed if req.seed >= 0 else random.randint(0, 2**32 - 1)

    t0 = time.time()
    try:
        images = await asyncio.to_thread(_generate, req, width, height, seed)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise HTTPException(507, "CUDA out of memory - lower resolution/n or set OFFLOAD_MODE=model")
    elapsed = time.time() - t0

    data = []
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        data.append({"b64_json": base64.b64encode(buf.getvalue()).decode()})
    log.info("Generated %d image(s) %dx%d in %.1fs seed=%d", len(images), width, height, elapsed, seed)
    return {"created": int(time.time()), "data": data, "seed": seed, "elapsed_seconds": round(elapsed, 2)}
