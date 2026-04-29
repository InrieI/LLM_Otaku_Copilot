from pathlib import Path
from subprocess import CalledProcessError, run
import audioop
import getpass
import json
import os
import wave

import numpy as np


FFMPEG_INFO = None
FFMPEG_PATH = None
try:
    import imageio_ffmpeg

    ffmpeg_path = Path(imageio_ffmpeg.get_ffmpeg_exe())
    if ffmpeg_path.exists():
        FFMPEG_PATH = ffmpeg_path
        FFMPEG_INFO = f"[system] ffmpeg: {ffmpeg_path}"
    else:
        FFMPEG_INFO = f"[warning] ffmpeg path missing: {ffmpeg_path}"
except Exception as exc:
    FFMPEG_INFO = f"[warning] ffmpeg setup failed: {exc}"

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

try:
    import whisper
except ImportError:
    whisper = None

try:
    from groq import Groq
except ImportError:
    Groq = None

DEFAULT_INPUT = Path("outputs") / "input.wav"
DEFAULT_MODEL_NAME = "medium"
DEFAULT_DEVICE = "cuda"
DEFAULT_COMPUTE_TYPE = "float16"
ROOT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.json"

DEFAULT_STT_CONFIG = {
    "provider": "local",
    "language": "zh",
    "local": {
        "model_name": DEFAULT_MODEL_NAME,
        "device": DEFAULT_DEVICE,
        "compute_type": DEFAULT_COMPUTE_TYPE,
    },
    "groq": {
        "model": "whisper-large-v3",
        "temperature": 0.0,
        "response_format": "verbose_json",
        "api_key": "",
    },
}


def _load_json(path, default_value):
    if not path.exists():
        return default_value
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default_value


def _save_json(path, data):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _load_root_config():
    data = _load_json(ROOT_CONFIG_PATH, {})
    if isinstance(data, dict):
        return data
    return {}


def _save_root_config(data):
    if not isinstance(data, dict):
        data = {}
    _save_json(ROOT_CONFIG_PATH, data)


def _normalize_stt_config(data):
    cfg = json.loads(json.dumps(DEFAULT_STT_CONFIG))
    if not isinstance(data, dict):
        return cfg

    provider = str(data.get("provider", cfg["provider"]))
    if provider not in {"local", "groq"}:
        provider = "local"
    cfg["provider"] = provider

    language = str(data.get("language", cfg["language"]))
    cfg["language"] = language if language else "zh"

    local = data.get("local", {}) if isinstance(data.get("local"), dict) else {}
    cfg["local"]["model_name"] = str(local.get("model_name", cfg["local"]["model_name"]))
    cfg["local"]["device"] = str(local.get("device", cfg["local"]["device"]))
    cfg["local"]["compute_type"] = str(local.get("compute_type", cfg["local"]["compute_type"]))

    groq = data.get("groq", {}) if isinstance(data.get("groq"), dict) else {}
    cfg["groq"]["model"] = str(groq.get("model", cfg["groq"]["model"]))
    try:
        cfg["groq"]["temperature"] = float(groq.get("temperature", cfg["groq"]["temperature"]))
    except (TypeError, ValueError):
        cfg["groq"]["temperature"] = 0.0
    cfg["groq"]["response_format"] = str(
        groq.get("response_format", cfg["groq"]["response_format"])
    )
    cfg["groq"]["api_key"] = str(groq.get("api_key", cfg["groq"]["api_key"]))

    return cfg


def _mask_key(key):
    if not key:
        return "(未设置)"
    if len(key) <= 10:
        return f"{key[:2]}****"
    return f"{key[:4]}****{key[-4:]}"


def _prompt_with_default(label, default_value):
    value = input(f"{label} (回车使用默认: {default_value})\n> ").strip()
    return value if value else default_value


def load_stt_config(output_dir=Path("outputs")):
    _ = output_dir
    root_config = _load_root_config()

    raw = root_config.get("stt")
    if not isinstance(raw, dict):
        raw = DEFAULT_STT_CONFIG

    config = _normalize_stt_config(raw)

    root_config["stt"] = config
    _save_root_config(root_config)
    return config



def load_audio_with_ffmpeg(file_path: Path, ffmpeg_exe: Path, sr: int = 16000):
    cmd = [
        str(ffmpeg_exe),
        "-nostdin",
        "-threads",
        "0",
        "-i",
        str(file_path),
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sr),
        "-",
    ]
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as exc:
        raise RuntimeError(f"音频解码失败: {exc.stderr.decode()}") from exc

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def transcribe_audio(
    filename=DEFAULT_INPUT,
    language="zh",
    model_name=DEFAULT_MODEL_NAME,
    device=DEFAULT_DEVICE,
    compute_type=DEFAULT_COMPUTE_TYPE,
    stt_config=None,
):
    input_path = Path(filename).resolve()
    if not input_path.exists():
        print(f"[error] 文件不存在: {input_path}")
        return None

    if _is_audio_too_quiet(input_path):
        print("[system] 音频过短或过静，已忽略。")
        return None

    if stt_config is None:
        config = _normalize_stt_config(
            {
                "provider": "local",
                "language": language,
                "local": {
                    "model_name": model_name,
                    "device": device,
                    "compute_type": compute_type,
                },
            }
        )
    else:
        config = _normalize_stt_config(stt_config)

    provider = config["provider"]

    if provider == "groq":
        return _transcribe_with_groq(input_path, config)

    local_cfg = config["local"]
    local_language = config["language"]
    local_model_name = local_cfg["model_name"]
    local_device = local_cfg["device"]
    local_compute_type = local_cfg["compute_type"]

    if FFMPEG_INFO:
        print(FFMPEG_INFO)

    print("[system] 正在加载 Whisper 模型...")
    print("[info] 首次运行会自动下载模型文件（约 1.5GB 左右）")

    try:
        if WhisperModel is None:
            if whisper is None:
                raise RuntimeError("未安装 faster-whisper 或 whisper")
            print("[system] 使用 whisper 作为后备方案")
            model = whisper.load_model(local_model_name, device="cpu")
            print(f"[system] 开始转录: {input_path}")
            if FFMPEG_PATH is not None:
                audio = load_audio_with_ffmpeg(input_path, FFMPEG_PATH)
                result = model.transcribe(audio, language=local_language)
            else:
                result = model.transcribe(str(input_path), language=local_language)
            text = result.get("text", "").strip()
        else:
            try:
                print(
                    f"[system] 使用 faster-whisper: model={local_model_name}, device={local_device}, compute_type={local_compute_type}"
                )
                print("[system] 正在加载模型(首次可能需要下载，时间较久)...")
                model = WhisperModel(local_model_name, device=local_device, compute_type=local_compute_type)
            except Exception as exc:
                print(f"[warning] CUDA 初始化失败，已切换到 CPU int8: {exc}")
                model = WhisperModel(local_model_name, device="cpu", compute_type="int8")

            print("[system] 模型加载完成")
            print(f"[system] 开始转录: {input_path}")
            if FFMPEG_PATH is not None:
                audio = load_audio_with_ffmpeg(input_path, FFMPEG_PATH)
                segments, _info = model.transcribe(
                    audio,
                    language=local_language,
                    beam_size=5,
                    vad_filter=True,
                )
            else:
                segments, _info = model.transcribe(
                    str(input_path),
                    language=local_language,
                    beam_size=5,
                    vad_filter=True,
                )
            text = "".join(segment.text for segment in segments).strip()

        print("[system] 转录完成")
        print(f"[text] {text}")
        return text
    except Exception as exc:
        print(f"[error] 转录失败: {exc}")
        return None


def _transcribe_with_groq(input_path: Path, config):
    if Groq is None:
        print("[error] 未安装 groq，请先执行: pip install groq")
        return None

    groq_cfg = config.get("groq", {})
    api_key = os.getenv("GROQ_API_KEY", "").strip() or str(groq_cfg.get("api_key", "")).strip()
    if not api_key:
        print("[error] 未设置 Groq API Key。请设置环境变量 GROQ_API_KEY 或在 config.json 的 stt.groq.api_key 填写。")
        return None

    model = str(groq_cfg.get("model", "whisper-large-v3"))
    temperature = float(groq_cfg.get("temperature", 0.0))
    response_format = str(groq_cfg.get("response_format", "verbose_json"))
    language = str(config.get("language", "zh")).strip()

    try:
        print(f"[system] 使用 Groq STT: model={model}")
        client = Groq(api_key=api_key)
        with open(input_path, "rb") as file:
            kwargs = {
                "file": (input_path.name, file.read()),
                "model": model,
                "temperature": temperature,
                "response_format": response_format,
            }
            if language:
                kwargs["language"] = language
            transcription = client.audio.transcriptions.create(**kwargs)

        text = getattr(transcription, "text", "")
        if not text and isinstance(transcription, dict):
            text = str(transcription.get("text", ""))
        text = (text or "").strip()

        print("[system] 转录完成")
        print(f"[text] {text}")
        return text
    except Exception as exc:
        print(f"[error] Groq 转录失败: {exc}")
        return None


def _is_audio_too_quiet(input_path: Path, min_seconds: float = 0.35, min_rms: int = 200) -> bool:
    try:
        with wave.open(str(input_path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate() or 16000
            if frames <= 0:
                return True
            duration = frames / float(rate)
            if duration < min_seconds:
                return True
            data = wf.readframes(frames)
            if not data:
                return True
            rms = audioop.rms(data, wf.getsampwidth())
            return rms < min_rms
    except Exception as exc:
        print(f"[warning] 音频检测失败: {exc}")
        return False


if __name__ == "__main__":
    cfg = load_stt_config()
    transcribe_audio(stt_config=cfg)
