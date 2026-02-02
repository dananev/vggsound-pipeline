# VGGSound Sound Effects Filtering Pipeline

Filter VGGSound dataset to identify videos containing **only sound effects** (no music, no speech) and generate rich text descriptions.

## Overview

This pipeline processes VGGSound videos through multiple stages:

1. **Extraction**: Videos from tar.gz → WAV audio (16kHz mono)
2. **Label Pre-filtering**: Fast rejection of obvious music/speech labels
3. **Speech Detection**: Silero VAD for voice activity detection
4. **Music Detection**: CLAP zero-shot classification
5. **Audio Captioning**: Microsoft CLAP clapcap (batch processing)
6. **Multimodal Verification** (optional): Qwen2.5-Omni for uncertain cases

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  1. EXTRACTION        2. FILTERING           3. CAPTIONING    4. OUTPUT     │
│  ┌──────────────┐    ┌────────────────┐     ┌─────────────┐  ┌───────────┐ │
│  │ MP4 → WAV    │───▶│ Speech: Silero │────▶│ CLAP clapcap│─▶│  JSONL    │ │
│  │ (ffmpeg)     │    │ Music: CLAP    │     │ (captioning)│  │ + scores  │ │
│  └──────────────┘    └────────────────┘     └─────────────┘  └───────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Installation

### System Requirements

- Python 3.10+
- PyTorch 2.3+
- ffmpeg (for audio extraction)
- Supported platforms:
  - **Linux** (x86_64, aarch64) - CPU or CUDA GPU
  - **macOS** (Apple Silicon only) - CPU-based inference
  - **Windows** (x86_64) - CPU or CUDA GPU

> **Note**: Intel Macs (x86_64) are not supported as PyTorch dropped support for them.

### Using uv (recommended)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone <repo_url>
cd vggsound-pipeline

# Choose your backend:

# Apple Silicon Mac or Linux CPU-only
uv sync --extra cpu

# Linux with CUDA 12.1 (e.g., Colab T4, older GPUs)
uv sync --extra cuda

# Linux with CUDA 12.4 (newer GPUs)
uv sync --extra cuda

# Add dev tools (pytest, ruff)
uv sync --extra cpu --extra dev
```

### Using pip

```bash
# CPU-only (macOS or Linux)
pip install -e ".[cpu]"

# With CUDA 12.1
pip install -e ".[cuda]"

# With CUDA 12.4
pip install -e ".[cuda]"
```

## Quick Start

```bash
# Basic run with 100 samples
vggsound run vggsound_00.tar.gz vggsound.csv --sample-limit 100

# Full run with multimodal verification
vggsound run vggsound_00.tar.gz vggsound.csv --enable-multimodal

# View statistics
vggsound stats output/sfx_filtered.jsonl

# Validate with random samples
vggsound validate output/sfx_filtered.jsonl --sample-size 20
```

## CLI Reference

### `vggsound run`

Main pipeline command.

```bash
vggsound run INPUT_TAR CSV_PATH [OPTIONS]

Options:
  --output, -o PATH           Output JSONL path [default: output/sfx_filtered.jsonl]
  --skip-label-filter         Skip label-based pre-filtering, use ML only
  --enable-multimodal         Enable Qwen2.5-Omni for uncertain samples
  --batch-size, -b INT        Batch size for GPU inference [default: 8]
  --num-workers, -w INT       Workers for audio extraction [default: 4]
  --sample-limit, -n INT      Limit samples (for testing)
  --speech-threshold FLOAT    Max speech ratio [default: 0.1]
  --music-threshold FLOAT     Max music score [default: 0.3]
  --device TEXT               Device: auto, cuda, cpu, mps [default: auto]
  --resume                    Resume from checkpoint
```

### `vggsound stats`

Show statistics for pipeline output.

```bash
vggsound stats OUTPUT_JSONL
```

### `vggsound validate`

Random sampling for manual validation.

```bash
vggsound validate OUTPUT_JSONL [OPTIONS]

Options:
  --sample-size, -n INT       Number of samples [default: 50]
  --audio-dir PATH            Directory with WAV files for playback
```

### `vggsound extract-labels`

Extract and categorize VGGSound labels.

```bash
vggsound extract-labels CSV_PATH [OPTIONS]

Options:
  --output, -o PATH           Output JSON path
```

## Output Format

```jsonl
{"video_id": "abc123_000030", "audio_text_description": "The sound of a dog barking repeatedly...", "speech_score": 0.02, "music_score": 0.05, "confidence": "high", "original_label": "dog barking"}
```

Fields:
- `video_id`: Unique identifier (YouTube ID + start time)
- `audio_text_description`: Rich text description from Qwen2-Audio
- `speech_score`: Detected speech ratio (0-1)
- `music_score`: Music probability score (0-1)
- `confidence`: Classification confidence level
- `original_label`: VGGSound category label

## Models Used

| Component | Model | Purpose |
|-----------|-------|---------|
| Speech Detection | Silero VAD | Fast CPU-based voice detection |
| Music Detection | LAION CLAP | Zero-shot audio classification |
| Captioning | Microsoft CLAP clapcap | Audio captioning with batch support |
| Multimodal | Qwen2.5-Omni-7B | Video+audio verification (optional) |

> **Note**: Microsoft CLAP clapcap was chosen for captioning - it's faster than LLM-based approaches
> and compatible with modern transformers. See [IMPLEMENTATION_DETAILS.md](IMPLEMENTATION_DETAILS.md).

## Configuration

Environment variables (prefix with `VGGSOUND_`):

```bash
export VGGSOUND_SPEECH_THRESHOLD=0.1
export VGGSOUND_MUSIC_THRESHOLD=0.3
export VGGSOUND_BATCH_SIZE=8
```

## Google Colab Setup

Colab provides T4 GPUs with CUDA, so use the `cuda` extra:

```python
# Install uv (faster than pip)
!curl -LsSf https://astral.sh/uv/install.sh | sh
!source ~/.local/bin/env

# Clone and install with CUDA support
!git clone <repo_url>
%cd vggsound-pipeline
!uv sync --extra cuda

# Or use pip if you prefer
!pip install torch torchaudio --index-url https://download.pytorch.org/whl/cuda
!pip install -e ".[cuda]"

# Upload data and run
!vggsound run vggsound_00.tar.gz vggsound.csv --sample-limit 200
```

## Development

```bash
# Install dev dependencies
uv sync --extra cpu --extra dev

# Run tests
uv run pytest

# Lint and format
uv run ruff check .
uv run ruff format .
```

## Architecture Notes

### Why These Models?

1. **Silero VAD**: Extremely fast (<1ms/chunk), works great on CPU, no GPU needed
2. **LAION CLAP**: Zero-shot classification, no training needed, specifically trained on music+speech
3. **Microsoft CLAP clapcap**: Built-in captioning, compatible with transformers>=4.34, batch processing
4. **Qwen2.5-Omni**: When audio-only is uncertain, video context helps disambiguate

### Why CLAP clapcap over LLM-based captioning?

For production-scale data pipelines:
- **Compatibility**: Works with modern transformers (>=4.34), no dependency conflicts
- **Efficiency**: Batch processing support, reasonable memory usage
- **Quality**: Good descriptions without requiring 7B+ parameter LLMs

For eval sets requiring highest quality, consider Qwen3-Omni-30B-A3B-Captioner via vLLM on A100s.

### Confidence Levels

- **high**: Strong signal that audio is SFX (speech < 0.05, music < 0.15)
- **medium**: Meets thresholds but not definitive
- **low**: Uncertain, may need multimodal verification
- **rejected**: Clear speech or music detected
- **multimodal_verified**: Verified by video+audio analysis

### Memory Management

- Models loaded sequentially (not all at once)
- CLAP models are lightweight, run on any GPU including T4
- CUDA cache cleared between stages
- Checkpointing for long runs and resume support

> **Note**: The Installation section lists outdated dependencies. Run `uv sync` or check
> `pyproject.toml` for current requirements.

### Platform Notes

| Platform | Notes |
|----------|-------|
| CUDA (Linux/Windows) | Best performance, CoNeTTE uses ~2GB VRAM |
| MPS (Apple Silicon) | GPU-accelerated, good performance |
| CPU | Slower but functional, CoNeTTE is lightweight |

## License

MIT
