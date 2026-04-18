from pathlib import Path
import wave

import numpy as np

try:
    import winsound
except ImportError:
    winsound = None


def _clamp_volume(value):
    try:
        volume = float(value)
    except (TypeError, ValueError):
        return 1.0
    if volume < 0:
        return 0.0
    if volume > 1.0:
        return 1.0
    return volume


def play_audio(path):
    path = Path(path)
    if not path.exists():
        return

    if winsound is None:
        print(f"[warning] Audio saved: {path}")
        return

    winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)


def _scale_pcm(frames, sampwidth, volume):
    if sampwidth == 2:
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
        data = np.clip(data * volume, -32768, 32767).astype(np.int16)
        return data.tobytes()

    if sampwidth == 4:
        data = np.frombuffer(frames, dtype=np.int32).astype(np.float64)
        data = np.clip(data * volume, -2147483648, 2147483647).astype(np.int32)
        return data.tobytes()

    if sampwidth == 1:
        data = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) * volume + 128.0
        data = np.clip(data, 0, 255).astype(np.uint8)
        return data.tobytes()

    return frames


def apply_volume_wav(path, volume, output_path):
    path = Path(path)
    output_path = Path(output_path)
    volume = _clamp_volume(volume)

    if volume >= 0.999:
        return path

    try:
        with wave.open(str(path), "rb") as reader:
            params = reader.getparams()
            frames = reader.readframes(reader.getnframes())

        scaled_frames = _scale_pcm(frames, params.sampwidth, volume)

        with wave.open(str(output_path), "wb") as writer:
            writer.setparams(params)
            writer.writeframes(scaled_frames)

        return output_path
    except Exception:
        print("[warning] Failed to apply volume, playing original audio.")
        return path
