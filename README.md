# Turbo-Term

Generate images from text prompts directly in your terminal. Uses the [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) model with a terminal-native workflow: generate → view inline → vary → save. Works in Ghostty and other terminals supporting the Kitty graphics protocol.

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Download the model from Hugging Face (~32GB)
HF_XET_HIGH_PERFORMANCE=1 hf download Tongyi-MAI/Z-Image-Turbo

# 3. Start generating
uv run python src/cli.py
```

## Features

- **Terminal Viewer** - Generate and view images inline with instant hotkeys
- **Auto-save** - Images saved to `~/Pictures/Autogen` automatically
- **Variations** - Same prompt, different seeds with one keypress
- **Web UI** - Gradio-based interface with resolution controls
- **Optimized for Mac** - Apple Silicon (MPS) and CUDA support

## Requirements

- Python 3.12+
- Mac with Apple Silicon (M1/M2/M3/M4) or NVIDIA GPU
- ~32GB disk space for model download
- 16GB+ RAM recommended

## Installation

```bash
# Install dependencies
uv sync

# Or with pip
pip install -e .

# Download the model
HF_XET_HIGH_PERFORMANCE=1 hf download Tongyi-MAI/Z-Image-Turbo
```

## Usage

### Terminal Viewer (Recommended)

```bash
uv run python src/cli.py                    # 640x480 (fast)
uv run python src/cli.py -W 1024 -H 1024    # 1024x1024 (high quality)
uv run python src/cli.py -s 6               # Fewer steps (faster)
```

**Options:** `-W/--width`, `-H/--height`, `-s/--steps`

Interactive hotkeys:
- `[c]` Copy to clipboard
- `[d]` Delete image
- `[u]` Upscale (2x resolution, same seed, max 1024)
- `[v]` Variation (new seed)
- `[r]` Reproduce (same seed)
- `[m]` Show memory usage
- `[n]` New prompt
- `[q]` Quit

The UI shows generation time, RAM usage, and GPU memory with a visual bar.

Images auto-save to `~/Pictures/Autogen/` with prompt-based filenames.

### Web UI

```bash
uv run python src/ui.py           # Standard UI
uv run python src/ui_optimized.py # With optimization options
```

Open http://localhost:7860 in your browser.

### Simple Generation

```bash
uv run python src/generate.py
```

Saves a single image to `output.png`.

## Performance

### Apple Silicon (M4 Max)

Based on benchmarking with 1024x1024 images, 9 steps:

| Configuration          | Time  | Speedup |
| ---------------------- | ----- | ------- |
| Standard               | ~100s | 1.00x   |
| Torch Compile          | ~162s | 0.62x   |
| Int8 Quantization      | ~126s | 0.80x   |
| Compile + Quantization | ~97s  | 1.03x   |

**Key Findings:**

- torch.compile is NOT optimized for MPS yet (adds overhead)
- int8 quantization provides minimal benefit (~3%)
- Model is already well-optimized for Apple Silicon
- **For faster generation**: reduce resolution or decrease steps

### CUDA GPUs

Expected performance on NVIDIA hardware:

- **RTX 4090**: 5-15s per image
- **H100**: Sub-5s per image
- torch.compile provides 2-3x speedup on CUDA
- bfloat16 recommended for best quality/speed balance

## Configuration

### Generation Parameters

Edit these in the scripts or UI:

```python
width = 1024              # Image width (512-2048)
height = 1024             # Image height (512-2048)
num_inference_steps = 9   # Quality vs speed (4-16)
guidance_scale = 0.0      # Always 0.0 for Turbo models
seed = 43                 # Reproducibility (-1 for random)
```

### Seed Notes

- Some seeds (e.g., 42) may produce black images
- Use 43 or -1 for random generation
- Same seed = same image (reproducible)

## Optimization Tips

### Faster Generation

- **Lower resolution**: 768x768 is ~2x faster than 1024x1024
- **Fewer steps**: Try 6-7 steps (slight quality tradeoff)
- **Use CUDA GPU**: 5-10x faster than Apple Silicon

### Memory Management

- Attention slicing enabled by default (reduces VRAM)
- Lower resolution if running out of memory
- Close other applications

## Troubleshooting

### Black Images

- Try seed=43 instead of seed=42
- Ensure using bfloat16 dtype
- Check model downloaded correctly

### Slow Performance

- First run downloads ~32GB model (one-time)
- torch.compile first run is slow (compilation)
- Apple Silicon: 90-100s per image is expected
- Use CUDA GPU for faster generation

### Out of Memory

- Reduce resolution (try 768x768 or 512x512)
- Enable attention slicing (enabled by default)
- Close other applications

## Project Structure

```
turbo-term/
├── src/
│   ├── cli.py               # Terminal viewer with Kitty graphics
│   ├── generate.py          # Simple generation script
│   ├── ui.py                # Gradio web UI
│   └── ui_optimized.py      # Web UI with optimization options
├── scripts/
│   ├── debug_model.py       # Device/dtype testing
│   └── optimize_benchmark.py # Performance benchmarking
├── pyproject.toml
└── README.md
```

## Dependencies

- **diffusers** - Image generation pipeline (from GitHub)
- **torch** - Deep learning framework
- **torchao** - Quantization library
- **gradio** - Web UI framework
- **transformers** - Model utilities
- **accelerate** - Training/inference optimization

## Model Information

**Z-Image-Turbo** by Tongyi-MAI:

- Optimized for fast inference (8-9 steps)
- 1024x1024 native resolution
- Fits in 16GB VRAM
- ~32GB model size on disk

**Model Card:** https://huggingface.co/Tongyi-MAI/Z-Image-Turbo

## License

MIT License. See [LICENSE](LICENSE) for details.

Note: The Z-Image-Turbo model has its own license. See the [model card](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) for details.

## Acknowledgments

- [Tongyi-MAI](https://huggingface.co/Tongyi-MAI) for the Z-Image-Turbo model
- [HuggingFace Diffusers](https://github.com/huggingface/diffusers) for the pipeline
- [PyTorch](https://pytorch.org/) and [torchao](https://github.com/pytorch/ao) for optimization tools
