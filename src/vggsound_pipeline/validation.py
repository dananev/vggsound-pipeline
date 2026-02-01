"""Validation and statistics for pipeline output.

Provides tools to:
- Analyze output quality
- Show score distributions
- Random sample for manual review
- Generate reports
"""

import random
from collections import Counter
from pathlib import Path

import orjson


def load_jsonl(jsonl_path: Path) -> list[dict]:
    """Load JSONL file into list of dicts.

    Args:
        jsonl_path: Path to JSONL file

    Returns:
        List of parsed JSON objects
    """
    results = []
    with open(jsonl_path, "rb") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(orjson.loads(line))
    return results


def show_stats(jsonl_path: Path):
    """Display statistics about pipeline output.

    Shows:
    - Total samples
    - Score distributions
    - Confidence breakdown
    - Label distribution (if available)

    Args:
        jsonl_path: Path to output JSONL file
    """
    results = load_jsonl(jsonl_path)

    if not results:
        print("No results found in file.")
        return

    print(f"\n{'='*60}")
    print(f"VGGSound Pipeline Results: {jsonl_path.name}")
    print(f"{'='*60}\n")

    # Basic counts
    print(f"Total samples: {len(results)}")

    # Confidence distribution
    confidence_counts = Counter(r.get("confidence", "unknown") for r in results)
    print("\nConfidence levels:")
    for level, count in sorted(confidence_counts.items()):
        pct = 100 * count / len(results)
        print(f"  {level}: {count} ({pct:.1f}%)")

    # Score distributions
    speech_scores = [r.get("speech_score", 0) for r in results]
    music_scores = [r.get("music_score", 0) for r in results]

    print("\nSpeech scores:")
    print(f"  Min: {min(speech_scores):.4f}")
    print(f"  Max: {max(speech_scores):.4f}")
    print(f"  Mean: {sum(speech_scores)/len(speech_scores):.4f}")

    print("\nMusic scores:")
    print(f"  Min: {min(music_scores):.4f}")
    print(f"  Max: {max(music_scores):.4f}")
    print(f"  Mean: {sum(music_scores)/len(music_scores):.4f}")

    # Score histogram (text-based)
    print("\nSpeech score distribution:")
    _print_histogram(speech_scores)

    print("\nMusic score distribution:")
    _print_histogram(music_scores)

    # Sample captions
    print(f"\n{'='*60}")
    print("Sample captions (first 3):")
    print(f"{'='*60}")
    for r in results[:3]:
        print(f"\n[{r.get('video_id', 'unknown')}]")
        print(f"  Speech: {r.get('speech_score', 0):.3f}, Music: {r.get('music_score', 0):.3f}")
        print(f"  Caption: {r.get('audio_text_description', 'N/A')[:200]}...")


def _print_histogram(values: list[float], bins: int = 10):
    """Print a text-based histogram.

    Args:
        values: List of float values
        bins: Number of histogram bins
    """
    min_val, max_val = min(values), max(values)
    bin_width = (max_val - min_val) / bins if max_val > min_val else 1

    bin_counts = [0] * bins
    for v in values:
        bin_idx = min(int((v - min_val) / bin_width), bins - 1)
        bin_counts[bin_idx] += 1

    max_count = max(bin_counts)
    bar_width = 40

    for i, count in enumerate(bin_counts):
        bin_start = min_val + i * bin_width
        bin_end = bin_start + bin_width
        bar_len = int(bar_width * count / max_count) if max_count > 0 else 0
        bar = "█" * bar_len
        print(f"  [{bin_start:.2f}-{bin_end:.2f}]: {bar} ({count})")


def validate_output(
    jsonl_path: Path,
    sample_size: int = 50,
    audio_dir: Path | None = None,
):
    """Validate output with random sampling.

    Args:
        jsonl_path: Path to output JSONL
        sample_size: Number of samples to review
        audio_dir: Optional directory with audio files for playback
    """
    results = load_jsonl(jsonl_path)

    if not results:
        print("No results to validate.")
        return

    # Random sample
    sample_size = min(sample_size, len(results))
    sample = random.sample(results, sample_size)

    print(f"\n{'='*60}")
    print(f"Validation Sample ({sample_size} random samples)")
    print(f"{'='*60}")

    for i, r in enumerate(sample, 1):
        print(f"\n[{i}/{sample_size}] {r.get('video_id', 'unknown')}")
        print(f"  Speech: {r.get('speech_score', 0):.4f}")
        print(f"  Music: {r.get('music_score', 0):.4f}")
        print(f"  Confidence: {r.get('confidence', 'unknown')}")
        print(f"  Caption: {r.get('audio_text_description', 'N/A')}")

        if audio_dir:
            audio_path = audio_dir / f"{r.get('video_id', '')}.wav"
            if audio_path.exists():
                print(f"  Audio: {audio_path}")

        print("-" * 40)

    print("\nValidation complete. Review samples above for quality.")
    print("Consider listening to audio files if available.")


def generate_report(
    jsonl_path: Path,
    output_path: Path | None = None,
) -> dict:
    """Generate a detailed report of pipeline results.

    Args:
        jsonl_path: Path to output JSONL
        output_path: Optional path to save report JSON

    Returns:
        Report dict with statistics and analysis
    """
    results = load_jsonl(jsonl_path)

    if not results:
        return {"error": "No results found"}

    speech_scores = [r.get("speech_score", 0) for r in results]
    music_scores = [r.get("music_score", 0) for r in results]
    confidence_counts = Counter(r.get("confidence", "unknown") for r in results)

    report = {
        "total_samples": len(results),
        "confidence_distribution": dict(confidence_counts),
        "speech_scores": {
            "min": min(speech_scores),
            "max": max(speech_scores),
            "mean": sum(speech_scores) / len(speech_scores),
        },
        "music_scores": {
            "min": min(music_scores),
            "max": max(music_scores),
            "mean": sum(music_scores) / len(music_scores),
        },
        "sample_captions": [
            {
                "video_id": r.get("video_id"),
                "caption": r.get("audio_text_description"),
            }
            for r in results[:5]
        ],
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(orjson.dumps(report, option=orjson.OPT_INDENT_2))
        print(f"Report saved to {output_path}")

    return report
