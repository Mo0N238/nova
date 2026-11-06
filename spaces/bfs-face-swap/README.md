---
title: BFS Best Face Swap
emoji: 🎭
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
python_version: "3.12"
short_description: Head swap with BFS LoRAs on FLUX.2 Klein 4B
license: mit
models:
  - Alissonerdx/BFS-Best-Face-Swap
  - black-forest-labs/FLUX.2-klein-4B
tags:
  - face-swap
  - head-swap
  - lora
---

# BFS: Best Face Swap demo

Gradio + ZeroGPU demo for [Alissonerdx/BFS-Best-Face-Swap](https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap)
using the FLUX.2 Klein 4B head-swap LoRAs (V1 and V1.1 optional).

- **Picture 1**: target body/base image (pose, lighting, background are kept)
- **Picture 2**: reference face/head

Only use images of people who have given consent.
