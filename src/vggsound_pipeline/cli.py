"""Command-line interface for the VGGSound filtering pipeline."""

from pathlib import Path

import typer

app = typer.Typer(
    name="vggsound",
    help="VGGSound Sound Effects Filtering Pipeline",
    add_completion=False,
)


@app.command()
def run(
    input_tar: Path = typer.Argument(
        ...,
        help="Path to vggsound_XX.tar.gz archive",
        exists=True,
        readable=True,
    ),
    csv_path: Path = typer.Argument(
        ...,
        help="Path to vggsound.csv metadata file",
        exists=True,
        readable=True,
    ),
    output: Path = typer.Option(
        Path("output/sfx_filtered.jsonl"),
        "--output",
        "-o",
        help="Output JSONL file path",
    ),
    skip_label_filter: bool = typer.Option(
        False,
        "--skip-label-filter",
        help="Skip label-based pre-filtering, use ML only",
    ),
    enable_multimodal: bool = typer.Option(
        False,
        "--enable-multimodal",
        help="Enable multimodal verification for low-confidence samples",
    ),
    batch_size: int = typer.Option(
        8,
        "--batch-size",
        "-b",
        help="Batch size for GPU inference",
        min=1,
        max=64,
    ),
    num_workers: int = typer.Option(
        4,
        "--num-workers",
        "-w",
        help="Workers for parallel audio extraction",
        min=1,
        max=32,
    ),
    sample_limit: int | None = typer.Option(
        None,
        "--sample-limit",
        "-n",
        help="Limit number of samples (for testing/demo)",
        min=1,
    ),
    speech_threshold: float = typer.Option(
        0.1,
        "--speech-threshold",
        help="Max speech ratio to accept as SFX",
        min=0.0,
        max=1.0,
    ),
    music_threshold: float = typer.Option(
        0.3,
        "--music-threshold",
        help="Max music score to accept as SFX",
        min=0.0,
        max=1.0,
    ),
    cache_dir: Path = typer.Option(
        Path(".cache/vggsound"),
        "--cache-dir",
        help="Directory for caching extracted files",
    ),
    hf_cache_dir: Path = typer.Option(
        Path("/content/hf_cache"),
        "--hf-cache-dir",
        help="Directory for caching HuggingFace model weights",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Resume from checkpoint if available",
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="Device for inference: 'auto', 'cuda', 'cpu', or 'mps'",
    ),
) -> None:
    """Filter VGGSound dataset to extract sound effects only.

    This pipeline:
    1. Extracts videos from tar.gz archive
    2. Converts video audio to WAV (16kHz mono)
    3. Optionally pre-filters by VGGSound labels
    4. Detects speech using Silero VAD
    5. Detects music using CLAP model
    6. Generates captions using video-SALMONN-2+
    7. Outputs filtered results to JSONL

    Example:
        vggsound run vggsound_00.tar.gz vggsound.csv --sample-limit 100
    """
    from .config import PipelineConfig
    from .pipeline import run_pipeline

    config = PipelineConfig(
        speech_threshold=speech_threshold,
        music_threshold=music_threshold,
        batch_size=batch_size,
        num_workers=num_workers,
        cache_dir=cache_dir,
        hf_cache_dir=hf_cache_dir,
    )

    output.parent.mkdir(parents=True, exist_ok=True)

    run_pipeline(
        input_tar=input_tar,
        csv_path=csv_path,
        output_path=output,
        config=config,
        skip_label_filter=skip_label_filter,
        enable_multimodal=enable_multimodal,
        sample_limit=sample_limit,
        resume=resume,
        device=device,
    )


@app.command()
def validate(
    jsonl_path: Path = typer.Argument(
        ...,
        help="Path to output JSONL file",
        exists=True,
        readable=True,
    ),
    sample_size: int = typer.Option(
        50,
        "--sample-size",
        "-n",
        help="Number of random samples to validate",
        min=1,
        max=500,
    ),
    audio_dir: Path | None = typer.Option(
        None,
        "--audio-dir",
        help="Directory containing WAV files for playback",
    ),
) -> None:
    """Validate output quality with random sampling and statistics."""
    from .validation import validate_output

    validate_output(
        jsonl_path=jsonl_path,
        sample_size=sample_size,
        audio_dir=audio_dir,
    )


@app.command()
def stats(
    jsonl_path: Path = typer.Argument(
        ...,
        help="Path to output JSONL file",
        exists=True,
        readable=True,
    ),
) -> None:
    """Show statistics about the pipeline output."""
    from .validation import show_stats

    show_stats(jsonl_path)


@app.command()
def extract_labels(
    csv_path: Path = typer.Argument(
        ...,
        help="Path to vggsound.csv",
        exists=True,
    ),
    output: Path = typer.Option(
        Path("output/label_categories.json"),
        "--output",
        "-o",
        help="Output JSON file for label categorization",
    ),
) -> None:
    """Extract and categorize unique labels from VGGSound metadata."""
    from .label_filter import extract_and_categorize_labels

    extract_and_categorize_labels(csv_path, output)


if __name__ == "__main__":
    app()
