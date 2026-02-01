# VGGSound Sound Effects Filtering Pipeline

Filter VGGSound dataset to identify videos containing **only sound effects** (no music, no speech) and generate rich text descriptions.

## Overview

This pipeline processes VGGSound videos through multiple stages:

1. **Extraction**: Videos from tar.gz → WAV audio (16kHz mono)
2. **Label Pre-filtering**: Fast rejection of obvious music/speech labels
3. **Speech Detection**: Silero VAD for voice activity detection
4. **Music Detection**: CLAP zero-shot classification
5. **Audio Captioning**: Qwen2-Audio rich descriptions
6. **Multimodal Verification** (optional): Qwen2.5-Omni for uncertain cases

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  1. EXTRACTION        2. FILTERING           3. CAPTIONING    4. OUTPUT     │
│  ┌──────────────┐    ┌────────────────┐     ┌─────────────┐  ┌───────────┐ │
│  │ MP4 → WAV    │───▶│ Speech: Silero │────▶│ Qwen2-Audio │─▶│  JSONL    │ │
│  │ (ffmpeg)     │    │ Music: CLAP    │     │             │  │ + scores  │ │
│  └──────────────┘    └────────────────┘     └─────────────┘  └───────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Installation

### Using uv (recommended)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone <repo_url>
cd vggsound-pipeline
uv sync

# With GPU support
uv sync --extra gpu
```

### Using pip

```bash
pip install -e .

# With GPU support
pip install -e ".[gpu]"
```

### System Requirements

- Python 3.10+
- ffmpeg (for audio extraction)
- CUDA GPU recommended (16GB+ VRAM for captioning)

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
| Music Detection | CLAP (laion/larger_clap_music_and_speech) | Zero-shot audio classification |
| Captioning | Qwen2-Audio-7B-Instruct | Rich audio descriptions |
| Multimodal | Qwen2.5-Omni-7B | Video+audio verification |

## Configuration

Environment variables (prefix with `VGGSOUND_`):

```bash
export VGGSOUND_SPEECH_THRESHOLD=0.1
export VGGSOUND_MUSIC_THRESHOLD=0.3
export VGGSOUND_BATCH_SIZE=8
```

## Google Colab Setup

```python
# Install
!pip install torch torchaudio transformers accelerate bitsandbytes \
    typer pydantic orjson ffmpeg-python

# Clone repo
!git clone <repo_url>
%cd vggsound-pipeline
!pip install -e .

# Upload data and run
!vggsound run vggsound_00.tar.gz vggsound.csv --sample-limit 200
```

## Development

```bash
# Install dev dependencies
uv sync --extra dev

# Run tests
pytest

# Lint and format
ruff check .
ruff format .
```

## Architecture Notes

### Why These Models?

1. **Silero VAD**: Extremely fast (<1ms/chunk), works great on CPU, no GPU needed
2. **CLAP**: Zero-shot classification means no training needed; specifically trained on music+speech
3. **Qwen2-Audio**: SOTA audio understanding, fits in T4 GPU with 8-bit quantization
4. **Qwen2.5-Omni**: When audio-only is uncertain, video context helps disambiguate

### Confidence Levels

- **high**: Strong signal that audio is SFX (speech < 0.05, music < 0.15)
- **medium**: Meets thresholds but not definitive
- **low**: Uncertain, may need multimodal verification
- **rejected**: Clear speech or music detected
- **multimodal_verified**: Verified by video+audio analysis

### Memory Management

- Models loaded sequentially (not all at once)
- 8-bit quantization for large models
- CUDA cache cleared periodically
- Checkpointing for long runs

## License

MIT
