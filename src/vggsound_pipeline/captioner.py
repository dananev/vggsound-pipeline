"""Audio-visual captioning using video-SALMONN-2+.

video-SALMONN-2+ is an audio-visual LLM from ByteDance/Tsinghua that:
- Uses Qwen2.5-VL as the vision-language backbone
- Uses Whisper encoder for audio understanding
- Interleaves audio and video tokens at 1-second intervals
- Produces contextually-aware captions by understanding both modalities

The 3B model achieves SOTA on audio-visual benchmarks while being efficient.
"""

import copy
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Default prompt for audio-visual captioning
DEFAULT_PROMPT = (
    "Describe the sounds in this video in detail. "
    "Focus on what you hear: sound effects, ambient sounds, actions creating sounds. "
    "Be specific about the source of each sound based on what you see."
)


class Captioner:
    """Generate captions from video+audio using video-SALMONN-2+.

    Processes both video frames and audio simultaneously, producing captions
    that understand the visual context of sounds.

    Example:
        >>> captioner = Captioner(device="cuda")
        >>> caption = captioner.caption(Path("video.mp4"))
        "Basketball players dribbling on court with sneakers squeaking"
    """

    def __init__(
        self,
        model_id: str = "tsinghua-ee/video-SALMONN-2_plus_3B",
        device: str = "auto",
        cache_dir: str | None = None,
        use_flash_attn: bool = True,
    ):
        """Initialize captioner.

        Args:
            model_id: HuggingFace model ID or local path
            device: Device for inference ("auto", "cuda", "cpu")
            cache_dir: Directory for caching model weights
            use_flash_attn: Use flash attention 2 (requires flash-attn package)
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.use_flash_attn = use_flash_attn and device == "cuda"

        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.audio_processor = None

        # Video processing settings
        self.video_max_frames = 8
        self.video_min_frames = 4
        self.base_interval = 4  # Sample 1 frame every 4 seconds

        # Audio settings
        self.audio_sample_rate = 16000
        self.audio_chunk_seconds = 30

    def load_model(self) -> None:
        """Load video-SALMONN-2+ model and processors."""
        if self.model is not None:
            return

        print(f"Loading video-SALMONN-2+ ({self.model_id}) on {self.device}...")

        from transformers import AutoTokenizer, WhisperFeatureExtractor

        from .qwenvl.data.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast
        from .qwenvl.model.modeling_qwen2_5_vl import video_SALMONN2_plus

        attn_impl = "flash_attention_2" if self.use_flash_attn else "eager"

        self.model = video_SALMONN2_plus.from_pretrained(
            self.model_id,
            attn_implementation=attn_impl,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
        )
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            cache_dir=self.cache_dir,
            trust_remote_code=True,
        )

        self.image_processor = Qwen2VLImageProcessorFast.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            cache_dir=self.cache_dir,
        )

        self.audio_processor = WhisperFeatureExtractor.from_pretrained(
            "openai/whisper-large-v3",
            cache_dir=self.cache_dir,
        )

        print("video-SALMONN-2+ loaded successfully")

    def _load_audio(self, audio_path: Path) -> tuple[np.ndarray, int]:
        """Load audio file at 16kHz mono."""
        try:
            from torchcodec.decoders import AudioDecoder

            decoder = AudioDecoder(
                str(audio_path),
                sample_rate=self.audio_sample_rate,
                num_channels=1,
            )
            audio = decoder.get_all_samples()
            return audio.data.numpy().squeeze(0), self.audio_sample_rate
        except Exception:
            import soundfile as sf

            audio, sr = sf.read(str(audio_path))
            if len(audio.shape) == 2:
                audio = audio[:, 0]
            if sr != self.audio_sample_rate:
                import torchaudio.functional as F

                audio_tensor = torch.from_numpy(audio).float()
                audio_tensor = F.resample(audio_tensor, sr, self.audio_sample_rate)
                audio = audio_tensor.numpy()
            return audio, self.audio_sample_rate

    def _process_audio(self, audio: np.ndarray) -> tuple[torch.Tensor, list[int]]:
        """Process audio into Whisper spectrograms."""
        sr = self.audio_sample_rate

        if len(audio) < sr:
            audio = np.pad(audio, (0, sr - len(audio)), mode="constant")

        chunk_size = self.audio_chunk_seconds * sr
        audio_chunks = [audio[k : k + chunk_size] for k in range(0, len(audio), chunk_size)]

        spectrograms = [
            self.audio_processor(chunk, sampling_rate=sr, return_tensors="pt")[
                "input_features"
            ].squeeze(0)
            for chunk in audio_chunks
        ]

        audio_feature = torch.stack(spectrograms, dim=0)
        audio_lengths = [60] * len(spectrograms)

        return audio_feature, audio_lengths

    def _load_video_frames(self, video_path: Path) -> tuple[np.ndarray, list[int], float]:
        """Load video frames for processing."""
        try:
            from torchcodec.decoders import VideoDecoder

            decoder = VideoDecoder(str(video_path), device="cpu")
            total_frames = decoder.metadata.num_frames
            avg_fps = decoder.metadata.average_fps
            video_length = total_frames / avg_fps

            num_frames = round(video_length / self.base_interval)
            target_frames = min(max(num_frames, self.video_min_frames), self.video_max_frames)

            frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
            frame_idx = np.unique(frame_idx)

            frame_batch = decoder.get_frames_at(indices=frame_idx.tolist())
            frames = frame_batch.data.cpu().numpy()

            return frames, frame_idx.tolist(), video_length

        except Exception:
            from decord import VideoReader, cpu

            vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
            total_frames = len(vr)
            fps = vr.get_avg_fps()
            video_length = total_frames / fps

            avg_fps = max(round(fps * self.base_interval), 1)
            frame_idx = list(range(0, total_frames, round(avg_fps)))

            if len(frame_idx) > self.video_max_frames:
                frame_idx = np.linspace(
                    0, total_frames - 1, self.video_max_frames, dtype=int
                ).tolist()

            frames = vr.get_batch(frame_idx).asnumpy().transpose(0, 3, 1, 2)

            return frames, frame_idx, video_length

    def _process_video(
        self, frames: np.ndarray, frame_idx: list[int], video_length: float
    ) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
        """Process video frames using Qwen2VL image processor."""
        processor = copy.deepcopy(self.image_processor)

        video_max_frame_pixels = 61250
        if len(frame_idx) < self.video_max_frames:
            new_pixel = 0.95 * self.video_max_frames / len(frame_idx) * video_max_frame_pixels
        else:
            new_pixel = video_max_frame_pixels

        processor.max_pixels = new_pixel
        processor.min_pixels = 256 * 28 * 28
        processor.size["longest_edge"] = processor.max_pixels
        processor.size["shortest_edge"] = processor.min_pixels

        video_processed = processor.preprocess(images=None, videos=frames, return_tensors="pt")

        pixel_values = video_processed["pixel_values_videos"]
        grid_thw = video_processed["video_grid_thw"][0]

        fps = len(frame_idx) / video_length
        second_per_grid_ts = [self.image_processor.temporal_patch_size / fps] * len(grid_thw)

        return pixel_values, grid_thw, second_per_grid_ts

    def _build_prompt(self, question: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Build input prompt with video token."""
        prompt = f"<|im_start|>user\n<video>\n{question}<|im_end|>\n<|im_start|>assistant\n"
        tokens = self.tokenizer(prompt, return_tensors="pt", padding=True)
        return tokens["input_ids"], tokens["attention_mask"]

    def caption(
        self,
        video_path: Path,
        audio_path: Path | None = None,
        prompt: str | None = None,
        max_new_tokens: int = 256,
    ) -> str:
        """Generate a caption for a video with audio.

        Args:
            video_path: Path to video file (MP4, etc.)
            audio_path: Path to audio file (WAV). If None, extracts from video.
            prompt: Custom prompt for captioning
            max_new_tokens: Maximum tokens to generate

        Returns:
            Text description of the audio-visual content
        """
        self.load_model()

        if prompt is None:
            prompt = DEFAULT_PROMPT

        frames, frame_idx, video_length = self._load_video_frames(video_path)
        pixel_values, grid_thw, second_per_grid_ts = self._process_video(
            frames, frame_idx, video_length
        )

        audio_source = audio_path if audio_path else video_path
        audio, sr = self._load_audio(audio_source)
        audio = audio[: int(video_length * sr)]

        audio_feature, audio_lengths = self._process_audio(audio)
        input_ids, attention_mask = self._build_prompt(prompt)

        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        pixel_values = pixel_values.to(dtype).to(self.device)
        grid_thw = grid_thw.to(self.device)
        audio_feature = audio_feature.to(dtype).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values_videos=pixel_values,
                video_grid_thw=grid_thw,
                audio_feature=audio_feature,
                audio_lengths=audio_lengths,
                second_per_grid_ts=second_per_grid_ts,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        output_ids = outputs[0, input_ids.shape[1] :]
        caption = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        return caption.strip()

    def caption_batch(
        self,
        video_paths: list[Path],
        audio_paths: list[Path] | None = None,
        prompt: str | None = None,
        show_progress: bool = True,
    ) -> dict[Path, str]:
        """Generate captions for multiple videos.

        Args:
            video_paths: List of video file paths
            audio_paths: Optional list of corresponding audio paths
            prompt: Custom prompt for all captions
            show_progress: Show progress bar

        Returns:
            Dict mapping video paths to captions
        """
        self.load_model()
        results = {}

        if audio_paths is None:
            audio_paths = [None] * len(video_paths)

        items = list(zip(video_paths, audio_paths))
        if show_progress:
            items = tqdm(items, desc="Captioning")

        for video_path, audio_path in items:
            try:
                results[video_path] = self.caption(video_path, audio_path, prompt)
            except Exception as e:
                print(f"Error captioning {video_path}: {e}")
                results[video_path] = f"Error: {e}"

        return results
