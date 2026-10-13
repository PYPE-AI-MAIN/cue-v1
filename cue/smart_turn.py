"""Lightweight helper for Pipecat's Smart Turn v3.2 (smart-turn-v3.2-cpu.onnx, 8.3 MB INT8 Whisper-Tiny encoder).

Handles the two subtle audio-prep requirements needed for `with_voice(probs, p_voice)`:
1. 8.0s (128,000 samples @ 16 kHz) right-aligned windowing: audio shorter than 8.0s is left-padded with zeros so the
   current pause onset sits at the right edge of the window, then normalised to zero-mean, unit-variance.
2. Exact (1, 80, 800) Whisper mel-spectrogram extraction (`padding=0` so FeatureExtractor does not emit 801 frames).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

WINDOW_SECONDS = 8
SAMPLE_RATE = 16000
WINDOW_SAMPLES = WINDOW_SECONDS * SAMPLE_RATE  # 128,000 samples


def find_smart_turn_model(*, auto_download: bool = False) -> Path | None:
    """Locate `smart-turn-v3.2-cpu.onnx` in the local Hugging Face cache, or download it if `auto_download=True`."""
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub" / "models--pipecat-ai--smart-turn-v3" / "snapshots"
    if hf_cache.is_dir():
        for snap in sorted(hf_cache.iterdir(), reverse=True):
            onnx_file = snap / "smart-turn-v3.2-cpu.onnx"
            if onnx_file.is_file() and onnx_file.stat().st_size > 1_000_000:
                return onnx_file

    if auto_download:
        try:
            from huggingface_hub import hf_hub_download

            downloaded = hf_hub_download(repo_id="pipecat-ai/smart-turn-v3", filename="smart-turn-v3.2-cpu.onnx")
            if downloaded and Path(downloaded).is_file():
                return Path(downloaded)
        except Exception:
            pass
    return None


def prepare_audio_window(audio: "np.ndarray | list[float]", sample_rate: int = SAMPLE_RATE) -> "np.ndarray":
    """Format raw audio to 8.0 seconds (128,000 samples @ 16 kHz), right-aligned and zero-mean unit-variance."""
    import numpy as np

    arr = np.asarray(audio, dtype=np.float32).flatten()
    if arr.size == 0:
        return np.zeros(WINDOW_SAMPLES, dtype=np.float32)

    if sample_rate != SAMPLE_RATE and sample_rate > 0:
        duration = arr.size / float(sample_rate)
        target_len = max(1, int(round(duration * SAMPLE_RATE)))
        arr = np.interp(
            np.linspace(0.0, 1.0, target_len, endpoint=False),
            np.linspace(0.0, 1.0, arr.size, endpoint=False),
            arr,
        ).astype(np.float32)

    if arr.size > WINDOW_SAMPLES:
        arr = arr[-WINDOW_SAMPLES:]
    elif arr.size < WINDOW_SAMPLES:
        arr = np.pad(arr, (WINDOW_SAMPLES - arr.size, 0), mode="constant", constant_values=0.0)

    mean = float(np.mean(arr))
    std = float(np.std(arr))
    arr = (arr - mean) / std if std > 1e-6 else (arr - mean)
    return arr.astype(np.float32)


class SmartTurn:
    """Pipecat Smart Turn v3.2 ONNX scorer (~9 ms on CPU) returning `p_voice` in [0.0, 1.0] for `with_voice()`."""

    def __init__(self, model_path: str | Path | None = None, *, auto_download: bool = False) -> None:
        self.model_path = Path(model_path) if model_path else find_smart_turn_model(auto_download=auto_download)
        self._session = None
        self._extractor = None
        self._input_name = "input_features"
        if self.model_path and self.model_path.is_file():
            self._init_session()

    def _init_session(self) -> None:
        try:
            import onnxruntime as ort
            from faster_whisper.feature_extractor import FeatureExtractor

            so = ort.SessionOptions()
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            so.inter_op_num_threads = 1
            so.intra_op_num_threads = 2
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(str(self.model_path), sess_options=so,
                                                 providers=["CPUExecutionProvider"])
            inputs = self._session.get_inputs()
            if inputs:
                self._input_name = inputs[0].name
            self._extractor = FeatureExtractor(chunk_length=WINDOW_SECONDS)
        except Exception:
            self._session = None
            self._extractor = None

    @property
    def is_available(self) -> bool:
        return self._session is not None and self._extractor is not None

    def predict(self, audio: "np.ndarray | list[float]", sample_rate: int = SAMPLE_RATE) -> float:
        """Return acoustic probability `p_voice` in [0.0, 1.0] that the speaker has finished their turn."""
        if not self.is_available or audio is None:
            return 0.5
        import numpy as np

        arr = np.asarray(audio, dtype=np.float32).flatten()
        if arr.size < int(sample_rate * 0.15):
            return 0.5
        aligned = prepare_audio_window(arr, sample_rate=sample_rate)
        mel = np.asarray(self._extractor(aligned, padding=0, chunk_length=WINDOW_SECONDS), dtype=np.float32)
        if mel.ndim == 2:
            mel = np.expand_dims(mel, axis=0)
        if mel.shape[-1] > 800:
            mel = mel[:, :, :800]
        elif mel.shape[-1] < 800:
            mel = np.pad(mel, ((0, 0), (0, 0), (0, 800 - mel.shape[-1])), mode="constant")

        outputs = self._session.run(None, {self._input_name: mel})
        val = float(np.asarray(outputs[0]).squeeze())
        prob = 1.0 / (1.0 + math.exp(-val)) if (val < 0.0 or val > 1.0) else val
        return max(0.001, min(0.999, prob))
