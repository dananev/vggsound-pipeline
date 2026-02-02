"""Main pipeline orchestration.

Coordinates all stages of the VGGSound filtering pipeline:
1. Data extraction (videos from tar, audio from videos)
2. Label-based pre-filtering (optional)
3. Speech detection (Silero VAD)
4. Music detection (CLAP)
5. Audio-visual captioning (video-SALMONN-2+)
6. Multimodal verification (optional)
7. Output generation (JSONL)
"""

from dataclasses import dataclass
from pathlib import Path

import orjson
import torch
from tqdm import tqdm

from .config import PipelineConfig
from .extraction import (
    VideoMetadata,
    extract_audio_batch,
    extract_videos_from_tar,
    get_videos_in_tar,
    parse_vggsound_csv,
)
from .label_filter import categorize_label, filter_by_labels


@dataclass
class ProcessedSample:
    """Result for a single processed sample."""

    video_id: str
    audio_path: Path | None
    video_path: Path | None
    metadata: VideoMetadata
    speech_score: float | None = None
    music_score: float | None = None
    confidence: str | None = None
    caption: str | None = None
    label_category: str | None = None
    multimodal_result: dict | None = None
    error: str | None = None


def _caption_sample(sample: ProcessedSample, captioner) -> None:
    """Generate caption for a single sample, updating it in place."""
    if not sample.video_path:
        sample.caption = sample.metadata.label
        return

    try:
        caption = captioner.caption(
            video_path=sample.video_path,
            audio_path=sample.audio_path,
        )
        sample.caption = caption if caption else sample.metadata.label
    except Exception as e:
        sample.caption = sample.metadata.label
        sample.error = str(e)
        print(f"  Caption error for {sample.video_id}: {e}")


def _run_captioning(
    samples: list[ProcessedSample],
    device: str,
    cache_dir: str,
) -> None:
    """Run captioning on all accepted samples."""
    print(f"\n[7/7] Generating captions for {len(samples)} samples...")

    from .captioner import Captioner

    captioner = Captioner(device=device, cache_dir=cache_dir)

    try:
        captioner.load_model()
    except Exception as e:
        print(f"  Failed to load captioner: {e}")
        print("  Using original labels as fallback...")
        for sample in samples:
            sample.caption = sample.metadata.label
        return

    for sample in tqdm(samples, desc="Captioning"):
        _caption_sample(sample, captioner)

    if device == "cuda":
        torch.cuda.empty_cache()


def run_pipeline(
    input_tar: Path,
    csv_path: Path,
    output_path: Path,
    config: PipelineConfig,
    skip_label_filter: bool = False,
    enable_multimodal: bool = False,
    sample_limit: int | None = None,
    resume: bool = False,
    device: str = "auto",
) -> None:
    """Run the full VGGSound filtering pipeline.

    Args:
        input_tar: Path to vggsound_XX.tar.gz
        csv_path: Path to vggsound.csv
        output_path: Path for output JSONL
        config: Pipeline configuration
        skip_label_filter: Skip label-based pre-filtering
        enable_multimodal: Enable multimodal verification for uncertain samples
        sample_limit: Limit number of samples (for testing)
        resume: Resume from checkpoint
        device: Device for inference
    """
    print("VGGSound Pipeline Starting")
    print("=" * 60)
    print(f"Input: {input_tar}")
    print(f"CSV: {csv_path}")
    print(f"Output: {output_path}")
    print(f"Device: {device}")
    print(f"Label filter: {'disabled' if skip_label_filter else 'enabled'}")
    print(f"Multimodal: {'enabled' if enable_multimodal else 'disabled'}")
    if sample_limit:
        print(f"Sample limit: {sample_limit}")
    print("=" * 60 + "\n")

    # Resolve device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"Using device: {device}")

    # Setup directories
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    video_dir = config.cache_dir / "videos"
    audio_dir = config.cache_dir / "audio"
    checkpoint_path = config.cache_dir / "checkpoint.json"

    # Load checkpoint if resuming
    processed_ids: set[str] = set()
    if resume and checkpoint_path.exists():
        with open(checkpoint_path, "rb") as f:
            checkpoint = orjson.loads(f.read())
            processed_ids = set(checkpoint.get("processed_ids", []))
        print(f"Resuming from checkpoint: {len(processed_ids)} samples already processed")

    # Step 1: Parse metadata
    print("\n[1/7] Parsing metadata...")
    all_metadata = parse_vggsound_csv(csv_path)
    print(f"  Total samples in CSV: {len(all_metadata)}")

    # Step 2: Scan tar archive
    print("\n[2/7] Scanning tar archive...")
    tar_videos = get_videos_in_tar(input_tar)
    print(f"  Videos in tar: {len(tar_videos)}")

    # Filter metadata to only include videos we have
    metadata_lookup = {m.sample_id: m for m in all_metadata}
    available_ids = {Path(v).stem for v in tar_videos} & set(metadata_lookup.keys())
    print(f"  Matched samples: {len(available_ids)}")

    if sample_limit:
        available_ids = set(list(available_ids)[:sample_limit])
        print(f"  Limited to: {len(available_ids)}")

    available_ids -= processed_ids
    print(f"  To process: {len(available_ids)}")

    # Step 3: Label-based pre-filtering
    candidates = [metadata_lookup[sid] for sid in available_ids]

    if not skip_label_filter:
        print("\n[3/7] Label-based pre-filtering...")
        sfx_candidates, rejected = filter_by_labels(candidates)
        print(f"  Rejected (music labels): {len(rejected['music'])}")
        print(f"  Rejected (speech labels): {len(rejected['speech'])}")
        print(f"  SFX candidates: {len(sfx_candidates)}")
        candidates = sfx_candidates
    else:
        print("\n[3/7] Skipping label-based filtering")

    if not candidates:
        print("No candidates to process after filtering.")
        return

    # Step 4: Extract videos
    print(f"\n[4/7] Extracting {len(candidates)} videos...")
    candidate_ids = {c.sample_id for c in candidates}
    video_paths = extract_videos_from_tar(input_tar, video_dir, video_ids=candidate_ids)
    video_path_map = {vp.stem: vp for vp in video_paths}
    print(f"  Extracted: {len(video_paths)} videos")

    # Step 5: Extract audio
    print("\n[5/7] Extracting audio...")
    audio_mapping = extract_audio_batch(
        video_paths,
        audio_dir,
        sample_rate=config.sample_rate,
        channels=config.audio_channels,
        num_workers=config.num_workers,
    )
    audio_path_map = {ap.stem: ap for ap in audio_mapping.values()}
    print(f"  Extracted: {len(audio_mapping)} audio files")

    # Step 6: ML-based filtering
    print("\n[6/7] ML-based filtering...")

    from .music_detector import MusicDetector
    from .speech_detector import SpeechDetector

    hf_cache = str(config.hf_cache_dir)
    speech_detector = SpeechDetector(device="cpu")  # VAD runs best on CPU
    music_detector = MusicDetector(device=device, cache_dir=hf_cache)

    audio_paths_to_process = [
        audio_path_map[c.sample_id] for c in candidates if c.sample_id in audio_path_map
    ]

    print("  Running speech detection...")
    speech_results = speech_detector.detect_speech_batch(audio_paths_to_process)

    print("  Running music detection...")
    music_results = music_detector.classify_batch(audio_paths_to_process)

    del music_detector
    if device == "cuda":
        torch.cuda.empty_cache()

    # Combine results and filter
    accepted_samples: list[ProcessedSample] = []
    low_confidence_samples: list[ProcessedSample] = []
    rejected_count = 0

    for candidate in candidates:
        if candidate.sample_id not in audio_path_map:
            continue

        audio_path = audio_path_map[candidate.sample_id]
        speech_score = speech_results.get(audio_path, {}).get("speech_ratio", 1.0)
        music_score = music_results.get(audio_path, {}).get("music_score", 1.0)
        confidence, is_accepted = config.get_confidence_level(speech_score, music_score)

        sample = ProcessedSample(
            video_id=candidate.sample_id,
            audio_path=audio_path,
            video_path=video_path_map.get(candidate.sample_id),
            metadata=candidate,
            speech_score=speech_score,
            music_score=music_score,
            confidence=confidence,
            label_category=categorize_label(candidate.label),
        )

        if confidence == "low" and enable_multimodal:
            low_confidence_samples.append(sample)
        elif is_accepted:
            accepted_samples.append(sample)
        else:
            rejected_count += 1

    print(f"  Accepted: {len(accepted_samples)}")
    print(f"  Low confidence (for multimodal): {len(low_confidence_samples)}")
    print(f"  Rejected: {rejected_count}")

    # Step 6b: Multimodal verification (if enabled)
    if enable_multimodal and low_confidence_samples:
        print(f"\n[6b] Multimodal verification for {len(low_confidence_samples)} samples...")
        from .multimodal import MultimodalVerifier

        verifier = MultimodalVerifier(device=device)
        for sample in tqdm(low_confidence_samples, desc="Multimodal"):
            if not sample.video_path:
                continue
            try:
                result = verifier.verify(sample.video_path)
                sample.multimodal_result = result

                if result.get("has_music") is False and result.get("has_speech") is False:
                    sample.confidence = "multimodal_verified"
                    accepted_samples.append(sample)
                elif result.get("is_sfx") is True:
                    sample.confidence = "multimodal_verified"
                    accepted_samples.append(sample)
            except Exception as e:
                print(f"  Error: {e}")

        print(f"  Total accepted after multimodal: {len(accepted_samples)}")

    # Step 7: Captioning
    if accepted_samples:
        _run_captioning(accepted_samples, device, hf_cache)

    # Write output
    print(f"\nWriting output to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        for sample in accepted_samples:
            record = {
                "video_id": sample.video_id,
                "audio_text_description": sample.caption or "",
                "speech_score": round(sample.speech_score or 0, 4),
                "music_score": round(sample.music_score or 0, 4),
                "confidence": sample.confidence,
                "original_label": sample.metadata.label,
            }
            f.write(orjson.dumps(record) + b"\n")

    # Update checkpoint
    new_processed = processed_ids | {s.video_id for s in accepted_samples}
    with open(checkpoint_path, "wb") as f:
        f.write(orjson.dumps({"processed_ids": list(new_processed)}))

    # Summary
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print(f"Total processed: {len(candidates)}")
    print(f"Accepted as SFX: {len(accepted_samples)}")
    print(f"Output: {output_path}")
    print("=" * 60)
