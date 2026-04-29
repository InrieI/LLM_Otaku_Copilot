from __future__ import annotations

from pathlib import Path
import threading
import time

try:
    import numpy as np
    import openwakeword
    from openwakeword.model import Model
    from openwakeword.utils import download_models
except Exception:
    np = None
    openwakeword = None
    download_models = None
    Model = None

import pyaudio


class WakeWordListener:
    def __init__(
        self,
        model_name: str,
        threshold: float,
        cooldown_seconds: float,
        on_wake,
        sample_rate: int = 16000,
        chunk_size: int = 1024,
        device_index: int = None,
    ):
        if Model is None or np is None or openwakeword is None:
            raise ImportError("openwakeword and numpy are required for WakeWordListener")
        model_path = Path(str(model_name))
        if not model_path.exists():
            available = set(openwakeword.MODELS.keys())
            if model_name not in available:
                raise ValueError(
                    f"Unknown wakeword model '{model_name}'. Available: {', '.join(sorted(available))}"
                )
            if download_models is not None:
                download_models([model_name])
        self.model_name = model_name
        self.threshold = threshold
        self.cooldown_seconds = cooldown_seconds
        self.on_wake = on_wake
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.device_index = device_index
        self._stop_event = threading.Event()
        self._thread = None
        self._last_trigger = 0.0

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(f"[system] Wakeword listener started ({self.model_name}).")

    def stop(self):
        self._stop_event.set()

    def _run(self):
        model_key = self.model_name
        model_path = Path(str(model_key))
        if model_path.exists():
            model_key = str(model_path)
        model = Model(wakeword_models=[model_key])

        while not self._stop_event.is_set():
            audio = pyaudio.PyAudio()
            stream = None
            try:
                stream_kwargs = {
                    "format": pyaudio.paInt16,
                    "channels": 1,
                    "rate": self.sample_rate,
                    "input": True,
                    "frames_per_buffer": self.chunk_size,
                }
                if self.device_index is not None:
                    stream_kwargs["input_device_index"] = int(self.device_index)
                stream = audio.open(**stream_kwargs)

                while not self._stop_event.is_set():
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    audio_chunk = np.frombuffer(data, dtype=np.int16)
                    prediction = model.predict(audio_chunk)
                    score = float(prediction.get(self.model_name, 0.0))

                    now = time.time()
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
                print(f"[warning] Wakeword listener error: {exc}")
                time.sleep(1.0)
            finally:
                try:
                    if stream is not None:
                        stream.stop_stream()
                        stream.close()
                except Exception:
                    pass
