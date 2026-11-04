FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    HF_HOME=/root/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv python3-dev curl ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && python3 -m venv $VIRTUAL_ENV

# PyTorch built against CUDA 12.9
RUN pip install --upgrade pip setuptools wheel \
 && pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu129

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY server.py .

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=1800s --retries=5 \
  CMD curl -fs http://localhost:8000/health || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
