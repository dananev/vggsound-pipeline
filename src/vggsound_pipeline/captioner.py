"""Audio captioning using Qwen2-Audio.

Qwen2-Audio-7B-Instruct is a multimodal model that can understand
and describe audio content with rich, natural language descriptions.

Memory optimization:
- CUDA: 8-bit quantization via bitsandbytes (fits in 16GB VRAM)
- MPS (Apple Silicon): fp16 on GPU
- CPU: fp32 fallback
"""

from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration


class AudioCaptioner:
    """Generate text descriptions for audio using Qwen2-Audio.

    The model produces rich, contextual descriptions of audio content,
    including details about sounds, their sources, and acoustic qualities.
    """

    MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"

    # Default prompt for captioning
    DEFAULT_PROMPT = (
        "Describe the sounds in this audio in detail. "
        "Focus on what you hear: the type of sounds, their characteristics, "
        "and any patterns or sequences. Be specific and descriptive."
    )

    def __init__(
        self,
        device: str = "auto",
        use_8bit: bool = True,
        prompt: str | None = None,
        cache_dir: str | None = None,
    ):
        """Initialize Qwen2-Audio model.

        Args:
            device: Device for inference ("auto", "cuda", "cpu", "mps")
            use_8bit: Use 8-bit quantization for memory efficiency
            prompt: Custom prompt for captioning (uses default if None)
            cache_dir: Directory to cache model weights (uses HF default if None)
        """
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.use_8bit = use_8bit and device == "cuda"  # 8-bit only works on CUDA
        self.prompt = prompt or self.DEFAULT_PROMPT
        self.cache_dir = cache_dir
        self.model = None
        self.processor = None
        self._sample_rate = 16000  # Qwen2-Audio expects 16kHz

    def load_model(self):
        """Load Qwen2-Audio model and processor."""
        if self.model is not None:
            return

        print(f"Loading Qwen2-Audio on {self.device}...")

        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID, cache_dir=self.cache_dir)

        # Configure quantization for memory efficiency
        if self.use_8bit:
            try:
                from transformers import BitsAndBytesConfig

                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                )
                self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                    self.MODEL_ID,
                    quantization_config=quantization_config,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    attn_implementation="sdpa",
                    cache_dir=self.cache_dir,
                )
            except ImportError:
                print("bitsandbytes not available, loading in full precision")
                self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                    self.MODEL_ID,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    attn_implementation="sdpa",
                    cache_dir=self.cache_dir,
                )
        else:
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                self.MODEL_ID,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                attn_implementation="sdpa",
                cache_dir=self.cache_dir,
            ).to(self.device)

    def caption(self, audio_path: Path, prompt: str | None = None) -> str:
        """Generate a text description for an audio file.

        Args:
            audio_path: Path to audio file
            prompt: Custom prompt (uses instance default if None)

        Returns:
            Text description of the audio content
        """
        self.load_model()

        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)

        # Resample if needed
        if sample_rate != self._sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self._sample_rate)
            waveform = resampler(waveform)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Prepare conversation format
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": str(audio_path)},
                    {"type": "text", "text": prompt or self.prompt},
                ],
            }
        ]

        # Apply chat template
        text = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )

        # Process inputs
        audios = [waveform.squeeze().numpy()]
        inputs = self.processor(
            text=text,
            audios=audios,
            sampling_rate=self._sample_rate,
            return_tensors="pt",
            padding=True,
        )

        # Move to device
        if not self.use_8bit:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate caption
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,  # Reduced for speed, still detailed
                do_sample=False,  # Deterministic output
            )

        # Decode response (skip input tokens)
        generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return response.strip()

    def caption_batch(
        self,
        audio_paths: list[Path],
        show_progress: bool = True,
    ) -> dict[Path, str]:
        """Generate captions for multiple audio files.

        Note: Processes one at a time to manage memory.
        For true batching, the model would need different memory management.

        Args:
            audio_paths: List of paths to audio files
            show_progress: Show progress bar

        Returns:
            Dict mapping audio paths to captions
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

            # Clear CUDA cache periodically to prevent OOM
            if self.device == "cuda" and len(results) % 10 == 0:
                torch.cuda.empty_cache()

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
        prompt: Custom prompt

    Returns:
        Text description
    """
    if captioner is None:
        captioner = AudioCaptioner()
    return captioner.caption(audio_path, prompt)
