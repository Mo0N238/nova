import spaces  # must be imported before torch / diffusers

import random
import time

import gradio as gr
import torch
from diffusers import Flux2KleinPipeline
from PIL import Image, ImageOps

BASE_MODEL = "black-forest-labs/FLUX.2-klein-4B"
LORA_REPO = "Alissonerdx/BFS-Best-Face-Swap"
LORAS = {
    "BFS Head V1 (Klein 4B)": ("bfs_head_v1_flux-klein_4b.safetensors", "bfs_v1"),
    "BFS Head V1.1 optional (Klein 4B)": ("bfs_head_v1.1_optional_flux-klein_4b.safetensors", "bfs_v1_1"),
}
DEFAULT_LORA = "BFS Head V1 (Klein 4B)"

# Official prompt from the BFS Flux 2 Klein guide (body-first input order).
DEFAULT_PROMPT = (
    "head_swap: start with Picture 1 as the base image, keeping its lighting, environment, and "
    "background. remove the head from Picture 1 completely and replace it with the head from "
    "Picture 2, strictly preserving the hair, eye color, and nose structure of Picture 2. copy the "
    "direction of the eye, head rotation, and micro-expressions from Picture 1. Keep the body, "
    "clothing, pose, framing, and background intact."
)
MAX_SEED = 2**31 - 1

pipe = Flux2KleinPipeline.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
for _label, (weight_name, adapter) in LORAS.items():
    pipe.load_lora_weights(LORA_REPO, weight_name=weight_name, adapter_name=adapter)
pipe.to("cuda")


def _fit_size(img: Image.Image, megapixels: float) -> tuple[int, int]:
    """Keep the target's aspect ratio at roughly `megapixels`, snapped to multiples of 16."""
    w, h = img.size
    scale = (megapixels * 1_000_000 / (w * h)) ** 0.5
    return max(256, int(w * scale) // 16 * 16), max(256, int(h * scale) // 16 * 16)


def _prep(img: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(img).convert("RGB")


def _duration(target, reference, lora, lora_scale, prompt, steps, megapixels, *args, **kwargs):
    return int(min(120, 20 + steps * 4 * max(1.0, megapixels)))


@spaces.GPU(duration=_duration)
def swap_head(
    target: Image.Image,
    reference: Image.Image,
    lora: str = DEFAULT_LORA,
    lora_scale: float = 1.0,
    prompt: str = DEFAULT_PROMPT,
    steps: int = 4,
    megapixels: float = 1.0,
    seed: int = 0,
    randomize_seed: bool = True,
    progress=gr.Progress(track_tqdm=True),
):
    """Swap the head in a target photo with the head from a reference photo using BFS (Best Face Swap) on FLUX.2 Klein 4B.

    Args:
        target: Picture 1, the base image whose body, pose, lighting and background are kept.
        reference: Picture 2, the person whose head/face should be transferred.
        lora: Which BFS Klein 4B LoRA to use.
        lora_scale: LoRA strength (start at 1.0).
        prompt: Edit instruction; the default is the official BFS prompt.
        steps: Number of inference steps (Klein 4B is distilled, 4 is typical).
        megapixels: Output resolution in megapixels; aspect ratio follows the target.
        seed: Random seed.
        randomize_seed: Pick a new random seed for each run.

    Returns:
        The swapped image, the seed used, and the generation time in seconds.
    """
    if target is None or reference is None:
        raise gr.Error("Upload both a target image (Picture 1) and a reference face (Picture 2).")
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    target, reference = _prep(target), _prep(reference)
    width, height = _fit_size(target, megapixels)
    pipe.set_adapters([LORAS[lora][1]], adapter_weights=[lora_scale])

    t0 = time.perf_counter()
    image = pipe(
        image=[target, reference],
        prompt=prompt or DEFAULT_PROMPT,
        width=width,
        height=height,
        num_inference_steps=int(steps),
        guidance_scale=1.0,
        generator=torch.Generator("cuda").manual_seed(int(seed)),
    ).images[0]
    return image, seed, round(time.perf_counter() - t0, 2)


with gr.Blocks(title="BFS Best Face Swap") as demo:
    gr.Markdown(
        """
# 🎭 BFS: Best Face Swap — FLUX.2 Klein 4B
Head swap with the [BFS LoRAs](https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap) by Alissonerdx on
[FLUX.2 Klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B).
**Picture 1** is the target (body, pose, background kept); **Picture 2** is the head/face to transfer.

⚠️ Only use photos of people who have given consent. Do not use images of public figures.
"""
    )
    with gr.Row():
        with gr.Column():
            with gr.Row():
                target = gr.Image(label="Picture 1 — target body", type="pil", height=360)
                reference = gr.Image(label="Picture 2 — reference face", type="pil", height=360)
            consent = gr.Checkbox(label="I have consent from the people in these photos", value=False)
            run = gr.Button("Swap head", variant="primary", interactive=False)
            with gr.Accordion("Advanced settings", open=False):
                lora = gr.Dropdown(list(LORAS), value=DEFAULT_LORA, label="LoRA")
                lora_scale = gr.Slider(0.0, 1.5, value=1.0, step=0.05, label="LoRA strength")
                prompt = gr.Textbox(DEFAULT_PROMPT, label="Prompt", lines=4)
                steps = gr.Slider(1, 12, value=4, step=1, label="Steps")
                megapixels = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="Output megapixels")
                with gr.Row():
                    seed = gr.Number(0, label="Seed", precision=0)
                    randomize_seed = gr.Checkbox(True, label="Randomize seed")
        with gr.Column():
            result = gr.Image(label="Result", format="png", height=560)
            with gr.Row():
                used_seed = gr.Number(label="Seed used", interactive=False)
                elapsed = gr.Number(label="Seconds", interactive=False)

    consent.change(lambda c: gr.Button(interactive=c), consent, run, api_visibility="private")
    run.click(
        swap_head,
        inputs=[target, reference, lora, lora_scale, prompt, steps, megapixels, seed, randomize_seed],
        outputs=[result, used_seed, elapsed],
        api_name="swap_head",
    )

if __name__ == "__main__":
    demo.launch(mcp_server=True, theme=gr.themes.Soft())
