# VGGSound SFX Pipeline - Implementation Details

## Architecture Overview

This pipeline filters VGGSound for sound effects (no music/speech) and generates captions at scale.

### Tiered Processing Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           DATA FLOW & SCALING                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   INPUT (TBs/day)                                                               │
│        │                                                                        │
│        ▼                                                                        │
│   ┌─────────────────┐                                                          │
│   │  TIER 1: FILTER │  CPU-heavy, 100s of workers                             │
│   │  Label + VAD    │  ~40% rejection → saves GPU cost                        │
│   └────────┬────────┘                                                          │
│            │ (~60% passes)                                                      │
│            ▼                                                                    │
│   ┌─────────────────┐                                                          │
│   │  TIER 2: CLAP   │  GPU classification, ~20 samples/s/GPU                  │
│   │  Music/Speech   │  ~50% rejection → quality gate                          │
│   └────────┬────────┘                                                          │
│            │ (~30% of original)                                                │
│            ▼                                                                    │
│   ┌─────────────────┐                                                          │
│   │  TIER 3: CAPTION│  Lightweight model, ~10 samples/s/GPU                   │
│   │  CoNeTTE        │  All accepted samples                                    │
│   └────────┬────────┘                                                          │
│            │                                                                    │
│            ▼                                                                    │
│   OUTPUT: sfx_filtered.jsonl                                                   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Stage-by-Stage Analysis

### Stage 1: Label Pre-Filtering

| Aspect | Details |
|--------|---------|
| **Purpose** | Reject obvious music/speech based on VGGSound labels |
| **SOTA Approach** | Fine-tuned BERT classifier on label semantics |
| **Our Approach** | Regex patterns (95+ rules) |
| **Why Practical** | Zero compute cost, ~40% rejection rate, no model loading |
| **Scalability** | O(1) per sample, CPU-only, infinitely parallelizable |
| **Tradeoff** | May miss edge cases, but false negatives are caught in Tier 2 |

### Stage 2: Audio Extraction

| Aspect | Details |
|--------|---------|
| **Purpose** | Convert MP4 → 16kHz mono WAV |
| **SOTA Approach** | GPU-accelerated decoding (NVIDIA Video Codec SDK) |
| **Our Approach** | ffmpeg with ProcessPoolExecutor |
| **Why Practical** | ffmpeg is ubiquitous, well-tested, handles edge cases |
| **Scalability** | ~50 files/s with 4 workers, I/O bound |
| **Tradeoff** | CPU-only extraction, but audio is small (10s clips) |

### Stage 3: Speech Detection

| Aspect | Details |
|--------|---------|
| **Purpose** | Detect voice/speech presence |
| **SOTA Approach** | Whisper-based VAD, Wav2Vec2 speech detection |
| **Our Approach** | Silero VAD (1.8MB, CPU) |
| **Why Practical** | <1ms/chunk, enterprise-grade accuracy, MIT license |
| **Scalability** | 1000+ samples/s on single CPU core |
| **Benchmark** | Silero VAD: 94.2% accuracy vs WebRTC: 87.1% |

### Stage 4: Music Detection

| Aspect | Details |
|--------|---------|
| **Purpose** | Classify audio as music/sfx/speech |
| **SOTA Approach** | PANNs ensemble, AudioSet classifier, fine-tuned CLAP |
| **Our Approach** | LAION CLAP zero-shot (music_and_speech variant) |
| **Why Practical** | No training needed, 4M sample pretraining, HuggingFace native |
| **Scalability** | ~20 samples/s/GPU, batch inference |
| **Benchmark** | ESC-50: 89.98%, GTZAN: 51% |

### Stage 5: Audio Captioning

| Aspect | Details |
|--------|---------|
| **Purpose** | Generate text descriptions for filtered audio |
| **SOTA Approach** | Qwen3-Omni-30B-A3B-Captioner (Sep 2025) |
| **Our Approach** | CoNeTTE (~100M params) |
| **Why Practical** | 10x faster, 1/40th memory, purpose-built for captioning |
| **Scalability** | ~10 samples/s/GPU vs ~1 sample/s for LLMs |
| **Quality** | SPIDEr 44% vs Qwen2-Audio ~55% (acceptable for pretraining data) |

---

## Scalability Calculations

### Throughput Requirements

```
TBs/day processing:
- 10s audio @ 16kHz mono ≈ 320KB
- 1 TB = ~3.2 million samples
- Target: 1 TB/day = ~37 samples/second sustained

Current pipeline throughput (single GPU):
- Tier 1 (CPU): 1000+ samples/s ✓
- Tier 2 (CLAP): ~20 samples/s ✓
- Tier 3 (CoNeTTE): ~10 samples/s ✓

Bottleneck: Captioning at ~10 samples/s
Solution: 4 GPUs in parallel = 40 samples/s > 37 samples/s ✓
```

### Slurm Job Structure

```bash
#!/bin/bash
#SBATCH --job-name=vggsound_pipeline
#SBATCH --array=0-99           # 100 shards
#SBATCH --gres=gpu:1           # 1 GPU per shard
#SBATCH --cpus-per-task=8      # For extraction parallelism
#SBATCH --mem=32G
#SBATCH --time=24:00:00

SHARD=$SLURM_ARRAY_TASK_ID
python -m vggsound_pipeline.run \
    --input-tar /data/vggsound_*.tar.gz \
    --shard $SHARD \
    --total-shards 100 \
    --output /output/shard_${SHARD}.jsonl
```

---

## Model Comparison (Feb 2026)

| Model | Params | VRAM | Speed | Quality | Use Case |
|-------|--------|------|-------|---------|----------|
| Qwen3-Omni-Captioner | 30B (3B active) | 80GB+ | ~1/s | SOTA | Eval sets |
| Qwen2-Audio-7B INT4 | 7B | ~5GB | ~1/s | High | Demo |
| Qwen2.5-Omni-3B | 3B | ~8GB | ~2/s | Med-High | Balance |
| **CoNeTTE** | 100M | ~2GB | **~10/s** | Medium | **Production** |
| video-SALMONN 2+ 3B | 3B | ~6GB | ~3/s | Med-High | Alternative |

**Recommendation**: CoNeTTE for production (10x throughput), Qwen-family for eval sets.

---

## Quality vs Throughput Tradeoff

```
Quality ────────────────────────────────────────────────────► Throughput
   │                                                              │
   │  Qwen3-Omni    Qwen2-Audio   CoNeTTE    Templates   Labels  │
   │  (SOTA)        (High)        (Medium)   (Low)       (Min)   │
   │  ~1/s          ~1/s          ~10/s      ~20/s       Instant │
   │                                                              │
   └──────────────────────────────────────────────────────────────┘

Industry practice (LAION-Audio-630K, WavCaps, AudioCaps):
- Bulk data: Template/keyword-based descriptions
- High-value: LLM-enhanced captions
- Eval sets: Human annotation
```

---

## Key Design Decisions

1. **Why Silero VAD over Whisper-VAD?**
   - 1000x faster (<1ms vs 100ms+ per chunk)
   - CPU-only (no GPU contention)
   - Comparable accuracy for our use case

2. **Why CLAP zero-shot over fine-tuned classifier?**
   - No training data needed
   - Generalizes to unseen categories
   - Single model for music + speech detection

3. **Why CoNeTTE over Qwen2-Audio?**
   - 10x throughput (10/s vs 1/s)
   - 1/40th memory (2GB vs 80GB)
   - Purpose-built for audio captioning task
   - Qwen for eval sets, CoNeTTE for scale

4. **Why tiered architecture?**
   - Early rejection saves GPU hours
   - Each tier uses optimal hardware (CPU/GPU)
   - Matches industry best practices (AudioLDM, WavCaps)

---

## References

- [VGGSound Dataset](https://huggingface.co/datasets/Loie/VGGSound) - 200K+ videos, 310 classes
- [Silero VAD Benchmarks](https://picovoice.ai/blog/best-voice-activity-detection-vad-2025/)
- [LAION CLAP](https://github.com/LAION-AI/CLAP) - Contrastive Language-Audio Pretraining
- [CoNeTTE](https://github.com/Labbeti/conette-audio-captioning) - Efficient Audio Captioning
- [Qwen3-Omni Technical Report](https://arxiv.org/abs/2509.17765) - Latest SOTA (Sep 2025)
- [AudioLDM Training](https://audioldm.github.io/) - Reference for data preparation
- [WavCaps](https://arxiv.org/abs/2303.17395) - ChatGPT-assisted weak labeling approach
