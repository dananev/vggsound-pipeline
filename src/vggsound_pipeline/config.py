"""Configuration and settings for the VGGSound pipeline.

Uses Pydantic Settings for type-safe configuration with environment variable support.
All thresholds and model parameters are centralized here for easy tuning.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class PipelineConfig(BaseSettings):
    """Main configuration for the VGGSound filtering pipeline.

    Attributes:
        speech_threshold: Maximum speech ratio (0-1) to consider audio as SFX.
            Audio with speech_ratio > this value is rejected.
        music_threshold: Maximum music score (0-1) to consider audio as SFX.
            Audio with music_score > this value is rejected.
        high_confidence_speech: Speech score below this is "high confidence no speech".
        high_confidence_music: Music score below this is "high confidence no music".
        reject_speech: Speech score above this triggers immediate rejection.
        reject_music: Music score above this triggers immediate rejection.
        sample_rate: Audio sample rate for processing (Hz).
        audio_channels: Number of audio channels (1=mono, 2=stereo).
        batch_size: Default batch size for GPU inference.
        num_workers: Default number of parallel workers.
        cache_dir: Directory for caching extracted audio and intermediate results.
    """

    # Filtering thresholds - these define the decision boundaries
    speech_threshold: float = Field(
        default=0.1,
        description="Max speech ratio to accept as SFX",
        ge=0.0,
        le=1.0,
    )
    music_threshold: float = Field(
        default=0.3,
        description="Max music score to accept as SFX",
        ge=0.0,
        le=1.0,
    )

    # High confidence thresholds - below these = definitely SFX
    high_confidence_speech: float = Field(
        default=0.05,
        description="Speech score below this is high confidence",
        ge=0.0,
        le=1.0,
    )
    high_confidence_music: float = Field(
        default=0.15,
        description="Music score below this is high confidence",
        ge=0.0,
        le=1.0,
    )

    # Rejection thresholds - above these = definitely NOT SFX
    reject_speech: float = Field(
        default=0.5,
        description="Speech score above this triggers rejection",
        ge=0.0,
        le=1.0,
    )
    reject_music: float = Field(
        default=0.5,
        description="Music score above this triggers rejection",
        ge=0.0,
        le=1.0,
    )

    # Audio processing settings
    sample_rate: int = Field(default=16000, description="Audio sample rate (Hz)")
    audio_channels: int = Field(default=1, description="Number of audio channels")

    # Processing settings
    batch_size: int = Field(default=8, description="Batch size for GPU inference")
    num_workers: int = Field(default=4, description="Workers for parallel extraction")

    # Paths
    cache_dir: Path = Field(
        default=Path(".cache/vggsound"),
        description="Directory for caching extracted files",
    )
    hf_cache_dir: Path = Field(
        default=Path(".cache/huggingface"),
        description="Directory for caching HuggingFace model weights",
    )

    model_config = {"env_prefix": "VGGSOUND_"}

    def get_confidence_level(
        self, speech_score: float, music_score: float
    ) -> tuple[str, bool]:
        """Determine confidence level and acceptance based on scores.

        Args:
            speech_score: Detected speech ratio (0-1)
            music_score: Detected music score (0-1)

        Returns:
            Tuple of (confidence_level, is_accepted) where:
            - confidence_level: "high", "low", or "rejected"
            - is_accepted: True if sample should be included in output
        """
        # High confidence rejection
        if speech_score > self.reject_speech or music_score > self.reject_music:
            return "rejected", False

        # High confidence acceptance
        if (speech_score < self.high_confidence_speech and
            music_score < self.high_confidence_music):
            return "high", True

        # Threshold-based acceptance
        if speech_score <= self.speech_threshold and music_score <= self.music_threshold:
            return "medium", True

        # Low confidence - needs multimodal verification or conservative rejection
        return "low", False


# Default configuration instance
default_config = PipelineConfig()
