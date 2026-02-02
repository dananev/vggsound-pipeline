"""Tests for VGGSound pipeline components."""

import tempfile
from pathlib import Path

import pytest

from vggsound_pipeline.config import PipelineConfig
from vggsound_pipeline.extraction import VideoMetadata, parse_vggsound_csv
from vggsound_pipeline.label_filter import categorize_label, filter_by_labels


class TestConfig:
    """Tests for PipelineConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PipelineConfig()
        assert config.speech_threshold == 0.1
        assert config.music_threshold == 0.3
        assert config.sample_rate == 16000
        assert config.batch_size == 8

    def test_confidence_high_accept(self):
        """Test high confidence acceptance."""
        config = PipelineConfig()
        confidence, accepted = config.get_confidence_level(0.01, 0.05)
        assert confidence == "high"
        assert accepted is True

    def test_confidence_rejected(self):
        """Test high confidence rejection."""
        config = PipelineConfig()
        confidence, accepted = config.get_confidence_level(0.8, 0.1)
        assert confidence == "rejected"
        assert accepted is False

    def test_confidence_low(self):
        """Test low confidence (uncertain) cases."""
        config = PipelineConfig()
        confidence, accepted = config.get_confidence_level(0.2, 0.25)
        assert confidence == "low"
        assert accepted is False


class TestLabelFilter:
    """Tests for label-based filtering."""

    def test_music_labels(self):
        """Test music label detection."""
        assert categorize_label("playing piano") == "music"
        assert categorize_label("playing guitar") == "music"
        assert categorize_label("orchestra") == "music"
        assert categorize_label("singing") == "music"
        assert categorize_label("beat boxing") == "music"

    def test_speech_labels(self):
        """Test speech label detection."""
        assert categorize_label("speech") == "speech"
        assert categorize_label("people speaking") == "speech"
        assert categorize_label("talking") == "speech"
        assert categorize_label("whispering") == "speech"

    def test_sfx_labels(self):
        """Test SFX label detection."""
        assert categorize_label("dog barking") == "sfx"
        assert categorize_label("car engine") == "sfx"
        assert categorize_label("rain") == "sfx"
        assert categorize_label("door closing") == "sfx"
        assert categorize_label("thunder") == "sfx"

    def test_filter_by_labels(self):
        """Test filtering metadata by labels."""
        metadata = [
            VideoMetadata("vid1", 0, "dog barking", "train"),
            VideoMetadata("vid2", 0, "playing piano", "train"),
            VideoMetadata("vid3", 0, "talking", "train"),
            VideoMetadata("vid4", 0, "thunder", "train"),
        ]
        sfx, rejected = filter_by_labels(metadata)

        assert len(sfx) == 2  # dog barking, thunder
        assert len(rejected["music"]) == 1  # playing piano
        assert len(rejected["speech"]) == 1  # talking


class TestVideoMetadata:
    """Tests for VideoMetadata."""

    def test_sample_id(self):
        """Test sample ID generation."""
        meta = VideoMetadata("abc123", 30, "dog barking", "train")
        assert meta.sample_id == "abc123_000030"

    def test_expected_filename(self):
        """Test expected filename generation."""
        meta = VideoMetadata("abc123", 30, "dog barking", "train")
        assert meta.expected_filename == "abc123_000030.mp4"


class TestCSVParsing:
    """Tests for CSV parsing."""

    def test_parse_csv(self):
        """Test parsing VGGSound CSV format."""
        # Create temporary CSV
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("vid1, 30, dog barking, train\n")
            f.write("vid2, 120, playing piano, test\n")
            csv_path = Path(f.name)

        try:
            metadata = parse_vggsound_csv(csv_path)
            assert len(metadata) == 2

            assert metadata[0].video_id == "vid1"
            assert metadata[0].start_time == 30
            assert metadata[0].label == "dog barking"
            assert metadata[0].split == "train"

            assert metadata[1].video_id == "vid2"
            assert metadata[1].start_time == 120
            assert metadata[1].label == "playing piano"
            assert metadata[1].split == "test"
        finally:
            csv_path.unlink()


# Integration tests that require model downloads are marked for skip by default
@pytest.mark.skip(reason="Requires model download")
class TestSpeechDetector:
    """Integration tests for speech detection."""

    def test_load_model(self):
        """Test model loading."""
        from vggsound_pipeline.speech_detector import SpeechDetector

        detector = SpeechDetector()
        detector.load_model()
        assert detector.model is not None


@pytest.mark.skip(reason="Requires model download")
class TestMusicDetector:
    """Integration tests for music detection."""

    def test_load_model(self):
        """Test model loading."""
        from vggsound_pipeline.music_detector import MusicDetector

        detector = MusicDetector(device="cpu")
        detector.load_model()
        assert detector.model is not None


@pytest.mark.skip(reason="Requires model download and video-SALMONN-2 repo")
class TestCaptioner:
    """Integration tests for video-audio captioning using video-SALMONN-2+."""

    def test_load_model(self):
        """Test model loading."""
        from vggsound_pipeline.captioner import Captioner

        captioner = Captioner(device="cpu")
        captioner.load_model()
        assert captioner.model is not None
