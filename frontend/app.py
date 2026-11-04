"""cflux4 frontend - Gradio UI (port 7860) talking to the backend on port 8000."""
import base64
import io
import os

import gradio as gr
import requests
from PIL import Image

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000").rstrip("/")
TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "1800"))

DEFAULT_PROMPT = (
    "A cinematic shot of a sleek futuristic sports car driving through "
    "a neon-lit cyberpunk city at night"
)


def backend_status() -> str:
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=5)
        j = r.json()
        msg = f"**Backend:** `{j.get('status')}` | model `{j.get('model')}` | mode `{j.get('mode')}`"
        if j.get("error"):
            msg += f"\n\n⚠️ {j['error']}"
        return msg
    except Exception as e:
        return f"**Backend:** unreachable at `{BACKEND_URL}` ({type(e).__name__})"


def generate(prompt, width, height, steps, guidance, max_seq, seed, n):
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a prompt.")
    payload = {
        "prompt": prompt,
        "size": f"{int(width)}x{int(height)}",
        "n": int(n),
        "num_inference_steps": int(steps),
        "guidance_scale": float(guidance),
        "max_sequence_length": int(max_seq),
        "seed": int(seed),
        "response_format": "b64_json",
    }
    try:
        r = requests.post(f"{BACKEND_URL}/v1/images/generations", json=payload, timeout=TIMEOUT)
    except requests.RequestException as e:
        raise gr.Error(f"Backend unreachable: {e}")
    if r.status_code != 200:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        raise gr.Error(f"Backend error {r.status_code}: {detail}")
    j = r.json()
    images = [Image.open(io.BytesIO(base64.b64decode(d["b64_json"]))) for d in j["data"]]
    info = f"Seed **{j['seed']}** · {j['elapsed_seconds']}s"
    return images, info, j["seed"]


with gr.Blocks(title="cflux4 · FLUX.1-dev") as demo:
    gr.Markdown("# cflux4 · FLUX.1-dev")
    status = gr.Markdown(backend_status())
    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(label="Prompt", value=DEFAULT_PROMPT, lines=4)
            with gr.Row():
                width = gr.Slider(256, 2048, value=1024, step=16, label="Width")
                height = gr.Slider(256, 2048, value=1024, step=16, label="Height")
            steps = gr.Slider(1, 100, value=50, step=1, label="Inference steps")
            guidance = gr.Slider(0.0, 10.0, value=3.5, step=0.1, label="Guidance scale")
            max_seq = gr.Slider(64, 512, value=512, step=64, label="Max sequence length")
            with gr.Row():
                seed = gr.Number(value=42, precision=0, label="Seed (-1 = random)")
                n = gr.Slider(1, 4, value=1, step=1, label="Images")
            with gr.Row():
                btn = gr.Button("Generate", variant="primary")
                refresh = gr.Button("Refresh status")
        with gr.Column(scale=1):
            gallery = gr.Gallery(label="Output", columns=2, height=640, format="png")
            info = gr.Markdown()
            last_seed = gr.Number(label="Last seed", interactive=False, precision=0)

    inputs = [prompt, width, height, steps, guidance, max_seq, seed, n]
    btn.click(generate, inputs, [gallery, info, last_seed])
    prompt.submit(generate, inputs, [gallery, info, last_seed])
    refresh.click(backend_status, None, status)
    demo.load(backend_status, None, status)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1, max_size=32).launch(
        server_name="0.0.0.0", server_port=7860, show_error=True
    )
