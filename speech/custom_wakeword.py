from __future__ import annotations

from pathlib import Path
import json
import threading
import time
import wave

import numpy as np
import pyaudio

try:
    import openwakeword
    from openwakeword.utils import AudioFeatures, download_models
except Exception:
    openwakeword = None
    AudioFeatures = None
    download_models = None


def _ensure_models_downloaded():
    if download_models is None:
        return
    try:
        download_models(["hey_jarvis"])
    except Exception as exc:
        print(f"[warning] Failed to download openwakeword models: {exc}")


def _create_feature_extractor(prefer_gpu: bool = True) -> AudioFeatures:
    if AudioFeatures is None:
        raise RuntimeError("openwakeword is not available for training")

    if prefer_gpu:
        try:
            return AudioFeatures(inference_framework="onnx", device="gpu")
        except Exception:
            pass
    try:
        return AudioFeatures(inference_framework="onnx", device="cpu")
    except Exception:
        return AudioFeatures(inference_framework="tflite", device="cpu")


def _load_wav_int16(path: Path, expected_sr: int = 16000) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            raise ValueError("Only mono 16-bit WAV is supported")
        sr = wf.getframerate()
        if sr != expected_sr:
            raise ValueError(f"Sample rate must be {expected_sr}, got {sr}")
        data = wf.readframes(wf.getnframes())
    return np.frombuffer(data, dtype=np.int16)


def _normalize_vector(vec: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(vec) + 1e-8
    return vec / denom


def _embed_clip(feature_extractor: AudioFeatures, samples: np.ndarray, window_seconds: float) -> np.ndarray:
    target_len = int(16000 * window_seconds)
    if samples.shape[0] > target_len:
        # For training clips that might be longer than the window, 
        # find the 1.5s segment with the highest energy (sum of absolute values).
        best_start = 0
        max_energy = -1.0
        # Use a sliding window with 50% overlap to find the "active" part
        step = target_len // 2
        for start in range(0, samples.shape[0] - target_len + 1, step):
            window = samples[start : start + target_len]
            energy = float(np.sum(np.abs(window)))
            if energy > max_energy:
                max_energy = energy
                best_start = start
        samples = samples[best_start : best_start + target_len]
    elif samples.shape[0] < target_len:
        pad = np.zeros(target_len - samples.shape[0], dtype=np.int16)
        samples = np.concatenate([samples, pad])

    embeddings = feature_extractor.embed_clips(samples[None, :], batch_size=1, ncpu=1)
    vec = embeddings.mean(axis=1).squeeze()
    return _normalize_vector(vec.astype(np.float32))


def dataset_stats(dataset_dir: Path) -> dict:
    pos_dir = dataset_dir / "positive"
    neg_dir = dataset_dir / "negative"
    pos_count = len(list(pos_dir.glob("*.wav"))) if pos_dir.exists() else 0
    neg_count = len(list(neg_dir.glob("*.wav"))) if neg_dir.exists() else 0
    return {"positive": pos_count, "negative": neg_count}


def train_custom_wakeword(
    dataset_dir: Path,
    output_path: Path,
    window_seconds: float = 1.5,
    prefer_gpu: bool = True,
) -> Path:
    if AudioFeatures is None:
        raise RuntimeError("openwakeword is not available for training")

    _ensure_models_downloaded()

    pos_dir = dataset_dir / "positive"
    neg_dir = dataset_dir / "negative"
    pos_files = sorted(pos_dir.glob("*.wav"))
    neg_files = sorted(neg_dir.glob("*.wav"))

    if len(pos_files) < 3:
        raise ValueError("Not enough positive samples (need at least 3)")
    if len(neg_files) < 3:
        raise ValueError("Not enough negative samples (need at least 3)")

    feature_extractor = _create_feature_extractor(prefer_gpu=prefer_gpu)

    pos_vectors = []
    for clip in pos_files:
        samples = _load_wav_int16(clip)
        pos_vectors.append(_embed_clip(feature_extractor, samples, window_seconds))

    neg_vectors = []
    for clip in neg_files:
        samples = _load_wav_int16(clip)
        neg_vectors.append(_embed_clip(feature_extractor, samples, window_seconds))

    centroid = _normalize_vector(np.mean(np.vstack(pos_vectors), axis=0))

    pos_scores = [float(np.dot(centroid, vec)) for vec in pos_vectors]
    neg_scores = [float(np.dot(centroid, vec)) for vec in neg_vectors]

    pos_mean = float(np.mean(pos_scores))
    neg_mean = float(np.mean(neg_scores))
    threshold = float((pos_mean + neg_mean) / 2.0)
    threshold = max(0.1, min(0.99, threshold))

    model = {
        "centroid": centroid.tolist(),
        "threshold": threshold,
        "window_seconds": window_seconds,
        "sample_rate": 16000,
        "created_at": time.time(),
        "positive_count": len(pos_vectors),
        "negative_count": len(neg_vectors),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(model, indent=2), encoding="utf-8")
    return output_path


def load_custom_model(model_path: Path) -> dict:
    data = json.loads(model_path.read_text(encoding="utf-8"))
    if "centroid" not in data:
        raise ValueError("Invalid wakeword model file")
    return data


class CustomWakeWordListener:
    def __init__(
        self,
        model_path: str,
        threshold: float | None,
        cooldown_seconds: float,
        on_wake,
        step_seconds: float = 0.25,
        device_index: int = None,
    ):
        if AudioFeatures is None:
            raise ImportError("openwakeword is required for CustomWakeWordListener")

        _ensure_models_downloaded()

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Custom wakeword model not found: {self.model_path}")

        model = load_custom_model(self.model_path)
        self.centroid = _normalize_vector(np.array(model["centroid"], dtype=np.float32))
        self.threshold = float(threshold if threshold is not None else model.get("threshold", 0.6))
        self.window_seconds = float(model.get("window_seconds", 1.5))
        self.sample_rate = int(model.get("sample_rate", 16000))
        self.step_seconds = step_seconds
        self.cooldown_seconds = float(cooldown_seconds)
        self.on_wake = on_wake
        self.device_index = device_index

        self._stop_event = False
        self._thread = None
        self._last_trigger = 0.0
        self._last_eval = 0.0

        self._feature_extractor = _create_feature_extractor(prefer_gpu=False)

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(f"[system] Custom wakeword listener started ({self.model_path.name}).")

    def stop(self):
        self._stop_event = True

    def _run(self):
        import collections
        chunk = int(self.sample_rate * 0.1)

        while not self._stop_event:
            buffer = collections.deque(maxlen=int(self.sample_rate * self.window_seconds))
            audio = pyaudio.PyAudio()
            stream = None
            try:
                stream_kwargs = {
                    "format": pyaudio.paInt16,
                    "channels": 1,
                    "rate": self.sample_rate,
                    "input": True,
                    "frames_per_buffer": chunk,
                }
                if self.device_index is not None:
                    stream_kwargs["input_device_index"] = int(self.device_index)
                stream = audio.open(**stream_kwargs)

                while not self._stop_event:
                    data = stream.read(chunk, exception_on_overflow=False)
                    samples = np.frombuffer(data, dtype=np.int16)
                    buffer.extend(samples.tolist())

                    if len(buffer) < buffer.maxlen:
                        continue

                    now = time.time()
                    if now - self._last_eval < self.step_seconds:
                        continue
                    self._last_eval = now

                    clip = np.array(buffer, dtype=np.int16)
                    
                    # Energy Gate: If the audio is too quiet (max amplitude < 500), 
                    # skip evaluation to prevent false triggers on background silence.
                    if np.max(np.abs(clip)) < 500:
                        continue

                    vec = _embed_clip(self._feature_extractor, clip, self.window_seconds)
                    score = float(np.dot(vec, self.centroid))
                    if score >= self.threshold and (now - self._last_trigger) >= self.cooldown_seconds:
                        self._last_trigger = now
                        stream.stop_stream()
                        stream.close()
                        stream = None
                        if self.on_wake:
                            self.on_wake()
                        time.sleep(self.cooldown_seconds)
                        break
            except Exception as exc:
                print(f"[warning] Custom wakeword listener error: {exc}")
                time.sleep(1.0)
            finally:
                try:
                    if stream is not None:
                        stream.stop_stream()
                        stream.close()
                except Exception:
                    pass
