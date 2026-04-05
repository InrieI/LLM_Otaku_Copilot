from pathlib import Path
from subprocess import CalledProcessError, run
import os

import numpy as np

# 自动设置 ffmpeg 路径（使用 imageio-ffmpeg 包）
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
    # 如果没装 imageio-ffmpeg，会用系统中的 ffmpeg
    FFMPEG_INFO = f"[warning] ffmpeg setup failed: {exc}"

import whisper

# 设置默认输入文件路径（从 outputs 文件夹里读取）
DEFAULT_INPUT = Path("outputs") / "input.wav"


def load_audio_with_ffmpeg(file_path: Path, ffmpeg_exe: Path, sr: int = 16000):
    """
    使用 ffmpeg 可执行文件直接解码音频为浮点数组
    
    参数：
    - file_path: 输入的 WAV 文件路径
    - ffmpeg_exe: ffmpeg 可执行文件的路径
    - sr: 采样率（默认 16000）
    
    返回: NumPy 数组，包含音频样本（范围 -1 到 1）
    """
    # 构造 ffmpeg 命令
    # -nostdin: 不从标准输入读取
    # -threads 0: 使用所有可用线程
    # -i: 输入文件
    # -f s16le: 输出格式为 16 bit PCM
    # -ac 1: 输出为单声道
    # -acodec pcm_s16le: 使用 PCM 编码
    # -ar: 采样率
    # -: 输出到标准输出（stdout）
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
        # 运行 ffmpeg 命令，获取输出的二进制数据
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as exc:
        # 如果命令失败，抛出异常并包含错误信息
        raise RuntimeError(f"音频解码失败: {exc.stderr.decode()}") from exc

    # 将二进制数据转换为 NumPy 数组：
    # 1. np.frombuffer(out, np.int16)：将字节流解释为 16 bit 整数
    # 2. .flatten()：展平为一维数组
    # 3. .astype(np.float32)：转换为 32 bit 浮点数
    # 4. / 32768.0：归一化到 -1.0 到 1.0 的范围（16 bit 整数最大值为 32767）
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def transcribe_audio(filename=DEFAULT_INPUT, language="zh", model_name="small"):
    """
    将 WAV 音频文件转换成文字（使用 OpenAI 的 Whisper 模型）
    
    参数:
    - filename: 要转录的音频文件路径
    - language: 音频语言（"zh" = 中文，"en" = 英文，等等）
    - model_name: Whisper 模型大小（"base" = 轻量级快速，"small"/"large" = 更准确但更慢）
    
    返回: 转录后的文字（字符串）或 None（失败）
    
    原理：
    Whisper 会"聆听"你的音频，然后将其转换为文字
    就像人类听一段录音然后打出来一样
    """
    input_path = Path(filename).resolve()

    # 检查文件是否存在
    if not input_path.exists():
        print(f"[error] 文件不存在: {input_path}")
        return None

    if FFMPEG_INFO:
        print(FFMPEG_INFO)

    print("[system] 正在加载 Whisper 模型...")
    print("[info] 首次运行会自动下载模型文件（约 461MB 左右）")

    try:
        # 加载 Whisper 模型到内存
        # device="cpu" 表示使用 CPU 运算（如果有 GPU 可以改成 "cuda"）
        model = whisper.load_model(model_name, device="cpu")
        
        print(f"[system] 开始转录: {input_path}")
        
        # 调用 Whisper 的 transcribe 方法进行转录
        # 返回的是一个字典，包含转录结果和其他信息
        if FFMPEG_PATH is not None:
            audio = load_audio_with_ffmpeg(input_path, FFMPEG_PATH)
            result = model.transcribe(audio, language=language)
        else:
            result = model.transcribe(str(input_path), language=language)
        
        # 从结果字典中提取转录的文字
        text = result.get("text", "").strip()
        
        print(f"[system] 转录完成")
        print(f"[text] {text}")  # 打印识别的文字
        
        return text
    except Exception as exc:
        print(f"[error] 转录失败: {exc}")
        return None


# "主程序"入口：当你直接运行这个文件时会执行
if __name__ == "__main__":
    # 直接转录默认文件
    transcribe_audio()
