"""Smart Turn v3.2 (pipecat-ai/smart-turn-v3) on the caller's audio: p(the turn is complete), from how it sounds.

Cue v1 reads words, not prosody. On LiveKit's eot-bench it ranks where it does combined with an audio turn model:
p(done) x sqrt(Smart Turn's score) (``cue.with_voice``). This runs Smart Turn locally on the CPU with onnxruntime and
numpy: pip install "cue-turn[voice] @ git+https://github.com/PYPE-AI-MAIN/cue-v1". The model (BSD-2, by Pipecat) is
downloaded from Hugging Face, not shipped here.

    from cue.smart_turn import SmartTurn
    model = SmartTurn()                   # downloads smart-turn-v3.2-cpu.onnx once, or SmartTurn("model.onnx")
    st = model.fresh()                    # one per call: the model is shared, the audio buffer is not
    st.push(pcm16k)                       # the caller's audio, as it arrives (16 kHz mono float or int16)
    p_done = with_voice(cue.end_of_turn(words, agent_said=question), st.predict())    # at each pause

Preprocessing follows the model's reference inference: the last 8 s of the turn, zeros in front when shorter,
the waveform normalised to zero mean and unit variance, then Whisper's 80-bin log-mel spectrogram (n_fft 400,
hop 160, Slaney mel scale, log10, floored 8 below the peak, (x + 4) / 4), 800 frames.
"""
from __future__ import annotations

import math
import threading
from collections import deque
from pathlib import Path

import numpy as np

RATE = 16000
SECONDS = 8
N_FFT, HOP, N_MELS = 400, 160, 80
REPO, FILE = "pipecat-ai/smart-turn-v3", "smart-turn-v3.2-cpu.onnx"


def _hz_to_mel(f: np.ndarray) -> np.ndarray:
    """Slaney's mel scale (librosa's default, Whisper's): linear below 1 kHz, logarithmic above."""
    f = np.asarray(f, dtype=np.float64)
    mel = 3.0 * f / 200.0
    log_part = f >= 1000.0
    mel[log_part] = 15.0 + np.log(f[log_part] / 1000.0) / (math.log(6.4) / 27.0)
    return mel


def _mel_to_hz(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    f = 200.0 * m / 3.0
    log_part = m >= 15.0
    f[log_part] = 1000.0 * np.exp((math.log(6.4) / 27.0) * (m[log_part] - 15.0))
    return f


def mel_filters(n_mels: int = N_MELS, n_fft: int = N_FFT, rate: int = RATE) -> np.ndarray:
    """Whisper's mel filter bank: triangular filters on the Slaney scale, Slaney-normalised. (n_mels, n_fft//2+1)."""
    fft_freqs = np.linspace(0.0, rate / 2.0, n_fft // 2 + 1)
    hz = _mel_to_hz(np.linspace(_hz_to_mel(np.array([0.0]))[0], _hz_to_mel(np.array([rate / 2.0]))[0], n_mels + 2))
    fdiff = np.diff(hz)
    ramps = hz[:, None] - fft_freqs[None, :]
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    weights = np.maximum(0.0, np.minimum(lower, upper))
    weights *= (2.0 / (hz[2:n_mels + 2] - hz[:n_mels]))[:, None]
    return weights.astype(np.float32)


_FILTERS = mel_filters()
_WINDOW = (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(N_FFT) / N_FFT)).astype(np.float64)   # periodic Hann


def features(audio: np.ndarray) -> np.ndarray:
    """The model's input for one turn: (1, 80, 800) float32."""
    x = np.asarray(audio, dtype=np.float64).reshape(-1)
    n = SECONDS * RATE
    x = x[-n:] if len(x) > n else np.pad(x, (n - len(x), 0))          # the last 8 s, zeros in front
    x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)                       # zero mean, unit variance
    x = np.pad(x, N_FFT // 2, mode="reflect")                          # centred frames
    frames = np.lib.stride_tricks.sliding_window_view(x, N_FFT)[::HOP]
    power = np.abs(np.fft.rfft(frames * _WINDOW, axis=1)) ** 2         # (801, 201)
    mel = _FILTERS.astype(np.float64) @ power.T                         # (80, 801)
    log = np.log10(np.maximum(mel, 1e-10))[:, :-1]                     # Whisper drops the last frame: 800
    log = np.maximum(log, log.max() - 8.0)
    return ((log + 4.0) / 4.0).astype(np.float32)[None]


class SmartTurn:
    """Smart Turn v3.2 over a rolling buffer of the caller's audio (the turn so far, at most 8 s)."""

    def __init__(self, model_path: str | Path | None = None, *, threads: int = 1) -> None:
        import onnxruntime as ort
        if model_path is None:
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download(REPO, FILE)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name
        self.buffer: deque[np.ndarray] = deque()
        self.samples = 0
        self._lock = threading.Lock()             # push() on the event loop, predict() in a worker thread

    def fresh(self) -> SmartTurn:
        """The same model with an empty buffer. Load the model once per process (e.g. in LiveKit's prewarm) and give
        each call its own ``fresh()`` copy: the model can be shared between calls, the buffer cannot."""
        other = object.__new__(SmartTurn)
        other.session, other.input = self.session, self.input
        other.buffer, other.samples, other._lock = deque(), 0, threading.Lock()
        return other

    def reset(self) -> None:
        """A new turn: forget the audio before it (call when the agent starts speaking)."""
        with self._lock:
            self.buffer.clear()
            self.samples = 0

    def push(self, pcm: np.ndarray) -> None:
        """16 kHz mono audio: float in [-1, 1] or int16."""
        a = np.asarray(pcm)
        a = a.astype(np.float32) / 32768.0 if a.dtype == np.int16 else a.astype(np.float32)
        with self._lock:
            self.buffer.append(a)
            self.samples += len(a)
            while self.buffer and self.samples - len(self.buffer[0]) >= SECONDS * RATE:
                self.samples -= len(self.buffer.popleft())

    def predict(self, audio: np.ndarray | None = None) -> float:
        """p(the turn is complete) for ``audio`` (16 kHz mono), or for the buffered turn."""
        if audio is None:
            with self._lock:
                chunks = list(self.buffer)
            if not chunks:
                return 0.5
            audio = np.concatenate(chunks)
        out = float(self.session.run(None, {self.input: features(audio)})[0].reshape(-1)[0])
        return out if 0.0 <= out <= 1.0 else 1.0 / (1.0 + math.exp(-out))   # a logits head: squash it
