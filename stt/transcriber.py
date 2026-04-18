from pathlib import Path
from subprocess import CalledProcessError, run

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

DEFAULT_INPUT = Path("outputs") / "input.wav"
DEFAULT_MODEL_NAME = "medium"
DEFAULT_DEVICE = "cuda"
DEFAULT_COMPUTE_TYPE = "float16"


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
):
    input_path = Path(filename).resolve()
    if not input_path.exists():
        print(f"[error] 文件不存在: {input_path}")
        return None

    if FFMPEG_INFO:
        print(FFMPEG_INFO)

    print("[system] 正在加载 Whisper 模型...")
    print("[info] 首次运行会自动下载模型文件（约 1.5GB 左右）")

    try:
        if WhisperModel is None:
            if whisper is None:
                raise RuntimeError("未安装 faster-whisper 或 whisper")
            print("[system] 使用 whisper 作为后备方案")
            model = whisper.load_model(model_name, device="cpu")
            print(f"[system] 开始转录: {input_path}")
            if FFMPEG_PATH is not None:
                audio = load_audio_with_ffmpeg(input_path, FFMPEG_PATH)
                result = model.transcribe(audio, language=language)
            else:
                result = model.transcribe(str(input_path), language=language)
            text = result.get("text", "").strip()
        else:
            try:
                print(
                    f"[system] 使用 faster-whisper: model={model_name}, device={device}, compute_type={compute_type}"
                )
                print("[system] 正在加载模型(首次可能需要下载，时间较久)...")
                model = WhisperModel(model_name, device=device, compute_type=compute_type)
            except Exception as exc:
                print(f"[warning] CUDA 初始化失败，已切换到 CPU int8: {exc}")
                model = WhisperModel(model_name, device="cpu", compute_type="int8")

            print("[system] 模型加载完成")
            print(f"[system] 开始转录: {input_path}")
            if FFMPEG_PATH is not None:
                audio = load_audio_with_ffmpeg(input_path, FFMPEG_PATH)
                segments, _info = model.transcribe(
                    audio,
                    language=language,
                    beam_size=5,
                    vad_filter=True,
                )
            else:
                segments, _info = model.transcribe(
                    str(input_path),
                    language=language,
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


if __name__ == "__main__":
    transcribe_audio()
