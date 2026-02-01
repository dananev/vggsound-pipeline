"""Multimodal verification using Qwen2.5-Omni.

For low-confidence samples where audio-only classification is uncertain,
Qwen2.5-Omni can analyze both video and audio together for better accuracy.

This module is optional and only used when --enable-multimodal is set.
"""

from pathlib import Path

import torch
from tqdm import tqdm


class MultimodalVerifier:
    """Verify uncertain samples using video + audio analysis.

    Qwen2.5-Omni-7B can process both visual and audio inputs,
    making it useful for disambiguating edge cases.
    """

    MODEL_ID = "Qwen/Qwen2.5-Omni-7B"

    # Prompt for verification
    VERIFICATION_PROMPT = """Analyze the video and audio carefully.

Answer these questions:
1. Does the audio contain music? (yes/no/uncertain)
2. Does the audio contain speech or someone talking? (yes/no/uncertain)
3. Is the audio primarily sound effects (environmental sounds, mechanical noises,
   animals, etc.)? (yes/no/uncertain)

Be precise about what you hear. If uncertain, say so."""

    def __init__(self, device: str = "auto", use_8bit: bool = True):
        """Initialize Qwen2.5-Omni model.

        Args:
            device: Device for inference
            use_8bit: Use 8-bit quantization for memory efficiency
        """
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.use_8bit = use_8bit and device == "cuda"
        self.model = None
        self.processor = None

    def load_model(self):
        """Load Qwen2.5-Omni model and processor.

        Note: This model is large (~14GB). On CUDA uses 8-bit quantization,
        on MPS/CPU uses fp16.
        """
        if self.model is not None:
            return

        print(f"Loading Qwen2.5-Omni on {self.device}...")
        if self.use_8bit:
            print("Using 8-bit quantization (~7GB VRAM)")
        else:
            print("Using fp16 (~14GB memory)")

        try:
            from transformers import AutoProcessor, Qwen2_5OmniForConditionalGeneration

            self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)

            if self.use_8bit:
                from transformers import BitsAndBytesConfig

                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
                self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                    self.MODEL_ID,
                    quantization_config=quantization_config,
                    device_map="auto",
                )
            else:
                self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                    self.MODEL_ID,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
        except ImportError as e:
            raise ImportError(
                "Qwen2.5-Omni requires additional dependencies. "
                "Install with: pip install qwen-omni-utils"
            ) from e

    def verify(self, video_path: Path) -> dict:
        """Verify if a video contains only sound effects.

        Args:
            video_path: Path to video file

        Returns:
            Dict with:
            - has_music: bool or None (uncertain)
            - has_speech: bool or None (uncertain)
            - is_sfx: bool or None (uncertain)
            - raw_response: Full model response
            - confidence: "high", "medium", or "low"
        """
        self.load_model()

        # Prepare conversation with video
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": str(video_path)},
                    {"type": "text", "text": self.VERIFICATION_PROMPT},
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
        inputs = self.processor(
            text=text,
            videos=[video_path],
            return_tensors="pt",
        )

        if not self.use_8bit:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
            )

        generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]

        return self._parse_response(response)

    def _parse_response(self, response: str) -> dict:
        """Parse model response to extract structured answers.

        Args:
            response: Raw model response text

        Returns:
            Structured dict with boolean/None values
        """
        response_lower = response.lower()

        def extract_answer(question_keywords: list[str]) -> bool | None:
            """Extract yes/no/uncertain from response."""
            # Look for patterns like "1. does... no" or "music? no"
            for keyword in question_keywords:
                if keyword in response_lower:
                    # Find the answer near this keyword
                    idx = response_lower.find(keyword)
                    context = response_lower[idx:idx + 100]

                    if "yes" in context.split("no")[0] if "no" in context else "yes" in context:
                        return True
                    if "no" in context:
                        return False
            return None  # Uncertain

        has_music = extract_answer(["music"])
        has_speech = extract_answer(["speech", "talking", "voice"])
        is_sfx = extract_answer(["sound effect", "environmental"])

        # Determine confidence based on how clear the answers are
        clear_answers = sum(1 for x in [has_music, has_speech, is_sfx] if x is not None)
        if clear_answers == 3:
            confidence = "high"
        elif clear_answers >= 2:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "has_music": has_music,
            "has_speech": has_speech,
            "is_sfx": is_sfx,
            "raw_response": response,
            "confidence": confidence,
        }

    def verify_batch(
        self,
        video_paths: list[Path],
        show_progress: bool = True,
    ) -> dict[Path, dict]:
        """Verify multiple videos.

        Args:
            video_paths: List of video file paths
            show_progress: Show progress bar

        Returns:
            Dict mapping paths to verification results
        """
        self.load_model()
        results = {}

        if show_progress:
            iterator = tqdm(video_paths, desc="Multimodal verification")
        else:
            iterator = video_paths

        for video_path in iterator:
            try:
                results[video_path] = self.verify(video_path)
            except Exception as e:
                print(f"Error verifying {video_path}: {e}")
                results[video_path] = {
                    "has_music": None,
                    "has_speech": None,
                    "is_sfx": None,
                    "error": str(e),
                    "confidence": "error",
                }

            # Clear cache periodically
            if self.device == "cuda" and len(results) % 5 == 0:
                torch.cuda.empty_cache()

        return results


def is_sfx_multimodal(video_path: Path, verifier: MultimodalVerifier | None = None) -> bool:
    """Quick helper to check if video is SFX using multimodal analysis.

    Args:
        video_path: Path to video file
        verifier: Optional pre-initialized verifier

    Returns:
        True if likely SFX, False otherwise
    """
    if verifier is None:
        verifier = MultimodalVerifier()

    result = verifier.verify(video_path)

    # Accept if no music, no speech, and is_sfx
    if result["has_music"] is False and result["has_speech"] is False:
        return True
    if result["is_sfx"] is True:
        return True

    return False
