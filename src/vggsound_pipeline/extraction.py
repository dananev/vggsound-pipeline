"""Video and audio extraction utilities.

Handles:
- Parsing VGGSound CSV metadata
- Extracting videos from tar.gz archives
- Converting video audio to WAV using ffmpeg
"""

import csv
import subprocess
import tarfile
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm


@dataclass
class VideoMetadata:
    """Metadata for a VGGSound video sample.

    Attributes:
        video_id: YouTube video ID
        start_time: Start time in seconds
        label: VGGSound label (e.g., "dog barking", "playing piano")
        split: Dataset split (train/test)
    """

    video_id: str
    start_time: int
    label: str
    split: str

    @property
    def sample_id(self) -> str:
        """Unique identifier combining video_id and start_time."""
        return f"{self.video_id}_{self.start_time:06d}"

    @property
    def expected_filename(self) -> str:
        """Expected filename in the tar archive."""
        return f"{self.video_id}_{self.start_time:06d}.mp4"


def parse_vggsound_csv(csv_path: Path) -> list[VideoMetadata]:
    """Parse VGGSound CSV file into metadata objects.

    The VGGSound CSV format is:
    video_id, start_time, label, split

    Args:
        csv_path: Path to vggsound.csv file

    Returns:
        List of VideoMetadata objects
    """
    metadata = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 4:
                video_id, start_time, label, split = row[0], row[1], row[2], row[3]
                metadata.append(
                    VideoMetadata(
                        video_id=video_id.strip(),
                        start_time=int(start_time.strip()),
                        label=label.strip(),
                        split=split.strip(),
                    )
                )
    return metadata


def get_videos_in_tar(tar_path: Path) -> set[str]:
    """Get set of video filenames in a tar archive without extracting.

    Args:
        tar_path: Path to tar.gz file

    Returns:
        Set of filenames (e.g., {"abc123_000030.mp4", ...})
    """
    videos = set()
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            if member.name.endswith(".mp4"):
                # Handle both flat and nested paths
                filename = Path(member.name).name
                videos.add(filename)
    return videos


def extract_videos_from_tar(
    tar_path: Path,
    output_dir: Path,
    video_ids: set[str] | None = None,
    limit: int | None = None,
) -> list[Path]:
    """Extract videos from tar.gz archive.

    Args:
        tar_path: Path to tar.gz file
        output_dir: Directory to extract videos to
        video_ids: Optional set of specific video IDs to extract.
            If None, extracts all videos.
        limit: Optional limit on number of videos to extract

    Returns:
        List of paths to extracted video files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted = []
    count = 0

    with tarfile.open(tar_path, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.name.endswith(".mp4")]

        # Filter by video_ids if provided
        if video_ids:
            members = [
                m for m in members if Path(m.name).stem in video_ids
            ]

        # Apply limit
        if limit:
            members = members[:limit]

        for member in tqdm(members, desc="Extracting videos"):
            # Extract to output directory, flattening any subdirectories
            filename = Path(member.name).name
            output_path = output_dir / filename

            # Skip if already extracted
            if output_path.exists():
                extracted.append(output_path)
                continue

            # Extract file
            member.name = filename  # Flatten path
            tar.extract(member, output_dir)
            extracted.append(output_path)

            count += 1
            if limit and count >= limit:
                break

    return extracted


def extract_audio_ffmpeg(
    video_path: Path,
    audio_path: Path,
    sample_rate: int = 16000,
    channels: int = 1,
) -> bool:
    """Extract audio from video using ffmpeg.

    Converts to WAV format suitable for ML models:
    - 16kHz sample rate (standard for speech/audio models)
    - Mono channel
    - 16-bit PCM

    Args:
        video_path: Path to input video file
        audio_path: Path for output WAV file
        sample_rate: Target sample rate (default 16000)
        channels: Number of channels (default 1 for mono)

    Returns:
        True if extraction succeeded, False otherwise
    """
    try:
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vn",  # No video
            "-acodec", "pcm_s16le",  # 16-bit PCM
            "-ar", str(sample_rate),  # Sample rate
            "-ac", str(channels),  # Channels
            "-y",  # Overwrite output
            str(audio_path),
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=60,
        )
        return result.returncode == 0 and audio_path.exists()
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"FFmpeg error for {video_path}: {e}")
        return False


def extract_audio_batch(
    video_paths: list[Path],
    output_dir: Path,
    sample_rate: int = 16000,
    channels: int = 1,
    num_workers: int = 4,
) -> dict[Path, Path]:
    """Extract audio from multiple videos in parallel.

    Args:
        video_paths: List of video file paths
        output_dir: Directory for output WAV files
        sample_rate: Target sample rate
        channels: Number of audio channels
        num_workers: Number of parallel workers

    Returns:
        Dict mapping video paths to their extracted audio paths.
        Failed extractions are not included.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    def process_video(video_path: Path) -> tuple[Path, Path | None]:
        audio_path = output_dir / f"{video_path.stem}.wav"
        if audio_path.exists():
            return video_path, audio_path
        success = extract_audio_ffmpeg(video_path, audio_path, sample_rate, channels)
        return video_path, audio_path if success else None

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_video, vp): vp for vp in video_paths}

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Extracting audio"
        ):
            video_path, audio_path = future.result()
            if audio_path:
                results[video_path] = audio_path

    return results


def iter_videos_streaming(
    tar_path: Path,
    metadata_lookup: dict[str, VideoMetadata],
    limit: int | None = None,
) -> Iterator[tuple[VideoMetadata, bytes]]:
    """Stream videos from tar without full extraction.

    Memory-efficient for large archives. Yields video data
    directly from the tar file.

    Args:
        tar_path: Path to tar.gz file
        metadata_lookup: Dict mapping sample_id to VideoMetadata
        limit: Optional limit on number of videos

    Yields:
        Tuples of (metadata, video_bytes)
    """
    count = 0
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.name.endswith(".mp4"):
                continue

            sample_id = Path(member.name).stem
            if sample_id not in metadata_lookup:
                continue

            f = tar.extractfile(member)
            if f is None:
                continue

            video_bytes = f.read()
            yield metadata_lookup[sample_id], video_bytes

            count += 1
            if limit and count >= limit:
                break
