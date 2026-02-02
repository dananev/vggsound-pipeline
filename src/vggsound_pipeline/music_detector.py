"""Music detection using CLAP (Contrastive Language-Audio Pretraining).

CLAP is a model that learns joint representations of audio and text,
enabling zero-shot audio classification by comparing audio embeddings
with text label embeddings.

We use the `laion/larger_clap_music_and_speech` model which is specifically
trained on music and speech data, making it ideal for distinguishing
sound effects from music/speech.
"""

from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm
from transformers import AutoProcessor, ClapModel


class MusicDetector:
    """Zero-shot music/speech detection using CLAP.

    Uses text prompts to classify audio into categories:
    - music
    - speech
    - sound effects
    - silence/ambient

    The model returns probability scores for each category.
    """

    # Default classification labels
    DEFAULT_LABELS = [
        "music playing",
        "someone speaking or talking",
        "sound effects and environmental sounds",
        "silence or ambient noise",
    ]

    # Model identifier
    MODEL_ID = "laion/larger_clap_music_and_speech"

    def __init__(self, device: str = "auto", labels: list[str] | None = None):
        """Initialize CLAP model.

        Args:
            device: Device for inference ("auto", "cuda", "cpu", "mps")
            labels: Custom classification labels (uses defaults if None)
        """
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.labels = labels or self.DEFAULT_LABELS
        self.model = None
        self.processor = None
        self._sample_rate = 48000  # CLAP expects 48kHz

    def load_model(self):
        """Load CLAP model and processor."""
        if self.model is not None:
            return

        print(f"Loading CLAP model on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        self.model = ClapModel.from_pretrained(self.MODEL_ID).to(self.device)
        # Set model to inference mode (not training)
        self.model.requires_grad_(False)

        # Pre-compute text embeddings for labels
        with torch.no_grad():
            text_inputs = self.processor(
                text=self.labels,
                return_tensors="pt",
                padding=True,
            )
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
            text_output = self.model.get_text_features(**text_inputs)
            # get_text_features returns BaseModelOutputWithPooling in newer transformers
            text_embeds = text_output.pooler_output if hasattr(text_output, "pooler_output") else text_output
            self._text_embeddings = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    def classify(self, audio_path: Path) -> dict[str, float]:
        """Classify audio using zero-shot CLAP.

        Args:
            audio_path: Path to audio file

        Returns:
            Dict mapping label names to probability scores
        """
        self.load_model()

        # Load and preprocess audio
        waveform, sample_rate = torchaudio.load(audio_path)

        # Resample to 48kHz if needed
        if sample_rate != self._sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self._sample_rate)
            waveform = resampler(waveform)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Process audio
        audio_inputs = self.processor(
            audios=waveform.squeeze().numpy(),
            sampling_rate=self._sample_rate,
            return_tensors="pt",
        )
        audio_inputs = {k: v.to(self.device) for k, v in audio_inputs.items()}

        # Get audio embedding
        with torch.no_grad():
            audio_features = self.model.get_audio_features(**audio_inputs)
            # Handle both tensor and object returns (newer transformers)
            if hasattr(audio_features, "pooler_output"):
                audio_features = audio_features.pooler_output
            audio_embedding = audio_features / audio_features.norm(dim=-1, keepdim=True)

            # Compute similarities with pre-computed text embeddings
            similarities = (audio_embedding @ self._text_embeddings.T).squeeze()

            # Convert to probabilities
            probs = torch.softmax(similarities * 100, dim=-1)  # Temperature scaling

        # Map to label names
        return {
            label: prob.item()
            for label, prob in zip(self.labels, probs)
        }

    def get_music_score(self, audio_path: Path) -> dict:
        """Get music and speech scores for filtering decisions.

        This is a convenience method that returns the scores most
        relevant for SFX filtering.

        Args:
            audio_path: Path to audio file

        Returns:
            Dict with:
            - music_score: Probability of music (0-1)
            - speech_score: Probability of speech (0-1)
            - sfx_score: Probability of sound effects (0-1)
            - all_scores: Full classification results
        """
        scores = self.classify(audio_path)

        # Extract relevant scores (handle different label formats)
        music_score = 0.0
        speech_score = 0.0
        sfx_score = 0.0

        for label, score in scores.items():
            label_lower = label.lower()
            if "music" in label_lower:
                music_score = max(music_score, score)
            elif "speech" in label_lower or "talk" in label_lower or "speak" in label_lower:
                speech_score = max(speech_score, score)
            elif "sound" in label_lower or "effect" in label_lower or "environment" in label_lower:
                sfx_score = max(sfx_score, score)

        return {
            "music_score": music_score,
            "speech_score": speech_score,
            "sfx_score": sfx_score,
            "all_scores": scores,
        }

    def classify_batch(
        self,
        audio_paths: list[Path],
        show_progress: bool = True,
    ) -> dict[Path, dict]:
        """Classify multiple audio files.

        Args:
            audio_paths: List of paths to audio files
            show_progress: Show progress bar

        Returns:
            Dict mapping audio paths to classification results
        """
        self.load_model()
        results = {}

        iterator = tqdm(audio_paths, desc="Music detection") if show_progress else audio_paths

        for audio_path in iterator:
            try:
                results[audio_path] = self.get_music_score(audio_path)
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                results[audio_path] = {
                    "music_score": 1.0,  # Conservative: assume music on error
                    "speech_score": 1.0,
                    "sfx_score": 0.0,
                    "error": str(e),
                }

        return results


def get_music_score(audio_path: Path, detector: MusicDetector | None = None) -> float:
    """Quick helper to get music score for a single file.

    Args:
        audio_path: Path to audio file
        detector: Optional pre-initialized detector

    Returns:
        Music score (0-1)
    """
    if detector is None:
        detector = MusicDetector()
    result = detector.get_music_score(audio_path)
    return result["music_score"]
