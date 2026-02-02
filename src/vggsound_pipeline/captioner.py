"""Audio captioning using CoNeTTE (lightweight, production-ready).

CoNeTTE is a purpose-built audio captioning model that is:
- Lightweight: ~100M parameters vs 7B+ for LLMs
- Fast: ~10 samples/second vs ~1 sample/second for LLMs
- Memory efficient: ~2GB VRAM vs 15-80GB for LLMs

This makes it ideal for production-scale audio annotation pipelines
where throughput matters more than SOTA quality.

For evaluation sets requiring highest quality captions,
consider Qwen3-Omni-30B-A3B-Captioner via vLLM on A100 GPUs.
"""

from pathlib import Path

import torch
from tqdm import tqdm


class AudioCaptioner:
    """Generate text descriptions for audio using CoNeTTE.

    CoNeTTE is a lightweight (~100M params) audio captioning model
    trained on AudioCaps, Clotho, WavCaps, and MACS datasets.
    Ideal for production-scale processing.

    Attributes:
        device: Device for inference (cuda, mps, cpu)
        task: Caption style ("audiocaps" or "clotho")
        model: Loaded CoNeTTE model instance
    """

    def __init__(
        self,
        device: str = "auto",
        task: str = "audiocaps",
        cache_dir: str | None = None,
        prompt: str | None = None,  # Ignored, kept for API compatibility
    ):
        """Initialize CoNeTTE captioner.

        Args:
            device: Device for inference ("auto", "cuda", "cpu", "mps")
            task: Caption style - "audiocaps" (short, factual) or "clotho" (longer, descriptive)
            cache_dir: Directory to cache model weights (unused, kept for API compatibility)
            prompt: Ignored - CoNeTTE uses task-based prompting, not custom prompts
        """
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.task = task
        self.cache_dir = cache_dir
        self.model = None

    def load_model(self):
        """Load CoNeTTE model and move to device."""
        if self.model is not None:
            return

        print(f"Loading CoNeTTE model on {self.device}...")

        from conette import CoNeTTEConfig, CoNeTTEModel

        config = CoNeTTEConfig.from_pretrained("Labbeti/conette")
        self.model = CoNeTTEModel.from_pretrained("Labbeti/conette", config=config)

        # Move to device if CUDA (CoNeTTE handles device placement internally for most cases)
        if self.device == "cuda":
            self.model = self.model.to(self.device)

        print(f"CoNeTTE loaded successfully (~2GB VRAM)")

    def caption(self, audio_path: Path, prompt: str | None = None) -> str:
        """Generate a text description for an audio file.

        Args:
            audio_path: Path to audio file (WAV, MP3, etc.)
            prompt: Ignored - CoNeTTE uses task-based prompting

        Returns:
            Text description of the audio content

        Note:
            CoNeTTE expects audio sampled at 32kHz but will auto-resample.
            Audio should be 1-30 seconds for best results.
        """
        self.load_model()

        try:
            outputs = self.model(str(audio_path), task=self.task)
            return outputs["cands"][0]
        except Exception as e:
            print(f"Error captioning {audio_path}: {e}")
            raise

    def caption_batch(
        self,
        audio_paths: list[Path],
        show_progress: bool = True,
    ) -> dict[Path, str]:
        """Generate captions for multiple audio files.

        Args:
            audio_paths: List of paths to audio files
            show_progress: Show progress bar

        Returns:
            Dict mapping audio paths to captions

        Note:
            Processes files one at a time to ensure consistent results.
            CoNeTTE is already fast (~10 samples/s), so batching
            provides minimal additional benefit.
        """
        self.load_model()
        results = {}

        iterator = tqdm(audio_paths, desc="Captioning") if show_progress else audio_paths

        for audio_path in iterator:
            try:
                results[audio_path] = self.caption(audio_path)
            except Exception as e:
                print(f"Error captioning {audio_path}: {e}")
                results[audio_path] = f"Error: {str(e)}"

        return results


def caption_audio(
    audio_path: Path,
    captioner: AudioCaptioner | None = None,
    prompt: str | None = None,
) -> str:
    """Quick helper to caption a single audio file.

    Args:
        audio_path: Path to audio file
        captioner: Optional pre-initialized captioner
        prompt: Ignored for CoNeTTE

    Returns:
        Text description
    """
    if captioner is None:
        captioner = AudioCaptioner()
    return captioner.caption(audio_path, prompt)
