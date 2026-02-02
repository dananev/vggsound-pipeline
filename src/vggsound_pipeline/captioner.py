"""Audio captioning using Microsoft CLAP (clapcap).

Microsoft CLAP includes a built-in audio captioning model that:
- Uses the 2023 CLAP encoders for audio understanding
- Generates descriptive captions without requiring an LLM
- Is compatible with transformers>=4.34.0
- Runs efficiently on GPU (~2-3GB VRAM)

This is faster than LLM-based approaches while maintaining good quality.
"""

from pathlib import Path

import torch
from tqdm import tqdm


class AudioCaptioner:
    """Generate text descriptions for audio using Microsoft CLAP clapcap.

    Microsoft's CLAP includes a captioning model that generates
    descriptive text from audio without requiring a separate LLM.

    Attributes:
        device: Device for inference (cuda, cpu)
        model: Loaded CLAP model instance
    """

    def __init__(
        self,
        device: str = "auto",
        cache_dir: str | None = None,
        prompt: str | None = None,  # Ignored, kept for API compatibility
        task: str | None = None,  # Ignored, kept for API compatibility
    ):
        """Initialize CLAP captioner.

        Args:
            device: Device for inference ("auto", "cuda", "cpu")
            cache_dir: Unused, kept for API compatibility
            prompt: Unused, kept for API compatibility
            task: Unused, kept for API compatibility
        """
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device
        self.use_cuda = device == "cuda"
        self.model = None

    def load_model(self):
        """Load Microsoft CLAP captioning model."""
        if self.model is not None:
            return

        print(f"Loading Microsoft CLAP captioner (clapcap) on {self.device}...")

        from msclap import CLAP

        self.model = CLAP(version="clapcap", use_cuda=self.use_cuda)

        print("CLAP captioner loaded successfully")

    def caption(self, audio_path: Path, prompt: str | None = None) -> str:
        """Generate a text description for an audio file.

        Args:
            audio_path: Path to audio file (WAV, MP3, etc.)
            prompt: Ignored - CLAP captioner doesn't use prompts

        Returns:
            Text description of the audio content
        """
        self.load_model()

        try:
            captions = self.model.generate_caption([str(audio_path)])
            return captions[0] if captions else ""
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
        """
        self.load_model()
        results = {}

        # Process in small batches for memory efficiency
        batch_size = 8
        paths_list = list(audio_paths)

        iterator = range(0, len(paths_list), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Captioning", total=len(paths_list) // batch_size + 1)

        for i in iterator:
            batch = paths_list[i : i + batch_size]
            batch_strs = [str(p) for p in batch]

            try:
                captions = self.model.generate_caption(batch_strs)
                for path, caption in zip(batch, captions):
                    results[path] = caption
            except Exception as e:
                print(f"Error captioning batch: {e}")
                # Fallback to individual processing
                for path in batch:
                    try:
                        results[path] = self.caption(path)
                    except Exception as e2:
                        print(f"Error captioning {path}: {e2}")
                        results[path] = f"Error: {str(e2)}"

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
        prompt: Ignored for CLAP captioner

    Returns:
        Text description
    """
    if captioner is None:
        captioner = AudioCaptioner()
    return captioner.caption(audio_path, prompt)
