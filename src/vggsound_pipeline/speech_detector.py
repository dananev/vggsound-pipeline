"""Speech detection using Silero VAD.

Silero VAD (Voice Activity Detection) is a lightweight, fast model
for detecting speech in audio. It runs on CPU with <1ms latency per chunk.

Key features:
- Pre-trained on large speech datasets
- Works well across languages and accents
- Returns timestamps of speech segments
- Calculates speech ratio (speech_duration / total_duration)
"""

from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm


class SpeechDetector:
    """Wrapper for Silero VAD model.

    The model analyzes audio and returns speech timestamps.
    We use these to calculate a speech_ratio metric.
    """

    def __init__(self, device: str = "cpu"):
        """Initialize Silero VAD model.

        Args:
            device: Device to run on ("cpu" recommended for VAD)
        """
        self.device = device
        self.model = None
        self.get_speech_timestamps = None
        self._sample_rate = 16000  # Silero requires 16kHz

    def load_model(self):
        """Load the Silero VAD model from torch.hub."""
        if self.model is not None:
            return

        # Load model from torch hub
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )
        self.model = model.to(self.device)
        self.get_speech_timestamps = utils[0]  # get_speech_timestamps function

    def detect_speech(self, audio_path: Path) -> dict:
        """Detect speech in an audio file.

        Args:
            audio_path: Path to WAV file (should be 16kHz mono)

        Returns:
            Dict with:
            - speech_ratio: Fraction of audio containing speech (0-1)
            - speech_segments: List of (start_sec, end_sec) tuples
            - total_duration: Total audio duration in seconds
        """
        self.load_model()

        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)

        # Resample if needed
        if sample_rate != self._sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self._sample_rate)
            waveform = resampler(waveform)
            sample_rate = self._sample_rate

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Get speech timestamps
        # Returns list of dicts: [{"start": sample_start, "end": sample_end}, ...]
        speech_timestamps = self.get_speech_timestamps(
            waveform.squeeze(),
            self.model,
            sampling_rate=sample_rate,
            threshold=0.5,  # Default confidence threshold
            min_speech_duration_ms=250,  # Min speech segment length
            min_silence_duration_ms=100,  # Min silence to split segments
        )

        # Calculate metrics
        total_samples = waveform.shape[1]
        total_duration = total_samples / sample_rate

        speech_samples = sum(
            ts["end"] - ts["start"] for ts in speech_timestamps
        )
        speech_duration = speech_samples / sample_rate
        speech_ratio = speech_duration / total_duration if total_duration > 0 else 0

        # Convert timestamps to seconds
        speech_segments = [
            (ts["start"] / sample_rate, ts["end"] / sample_rate)
            for ts in speech_timestamps
        ]

        return {
            "speech_ratio": speech_ratio,
            "speech_segments": speech_segments,
            "total_duration": total_duration,
            "speech_duration": speech_duration,
        }

    def detect_speech_batch(
        self,
        audio_paths: list[Path],
        show_progress: bool = True,
    ) -> dict[Path, dict]:
        """Detect speech in multiple audio files.

        Args:
            audio_paths: List of paths to WAV files
            show_progress: Show progress bar

        Returns:
            Dict mapping audio paths to detection results
        """
        self.load_model()
        results = {}

        iterator = tqdm(audio_paths, desc="Speech detection") if show_progress else audio_paths

        for audio_path in iterator:
            try:
                results[audio_path] = self.detect_speech(audio_path)
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                results[audio_path] = {
                    "speech_ratio": 1.0,  # Assume speech on error (conservative)
                    "speech_segments": [],
                    "total_duration": 0,
                    "error": str(e),
                }

        return results


def get_speech_ratio(audio_path: Path, detector: SpeechDetector | None = None) -> float:
    """Quick helper to get speech ratio for a single file.

    Args:
        audio_path: Path to audio file
        detector: Optional pre-initialized detector

    Returns:
        Speech ratio (0-1)
    """
    if detector is None:
        detector = SpeechDetector()
    result = detector.detect_speech(audio_path)
    return result["speech_ratio"]
