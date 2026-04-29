from pathlib import Path

import keyboard
import pyaudio
import time
import wave
import warnings

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

try:
    import webrtcvad
except Exception:
    webrtcvad = None

# 设置默认输出文件路径（在 outputs 文件夹里）
DEFAULT_OUTPUT = Path("outputs") / "input.wav"

# 全局状态字典：用来存放正在进行的录音信息
# 这样即使函数返回了，我们仍然能在另一个函数里继续使用这些数据
recording_state = {"stream": None, "frames": None, "audio": None, "is_recording": False}


def start_recording(
    filename=DEFAULT_OUTPUT,
    stop_hint="再次按当前热键停止并处理",
    vad_config=None,
    input_device_index=None,
):
    """
    开始录音（由主流程热键触发）
    
    原理：
    1. 打开麦克风
    2. 创建一个后台线程不断读取音频数据
    3. 将数据保存到内存中
    4. 等待调用 stop_recording() 来停止
    
    参数: filename - 录音要保存到的文件路径
    返回: True（成功） 或 False（失败）
    """
    output_path = Path(filename)
    # 确保 outputs 文件夹存在，不存在就创建
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 音频参数设置
    audio_format = pyaudio.paInt16          # 16 位整数格式（质量好）
    channels = 1                            # 单声道（mono）
    rate = 16000                            # 采样率：每秒 16000 个采样点

    audio = pyaudio.PyAudio()  # 创建 PyAudio 对象（麦克风管理器）

    vad_config = vad_config or {}
    vad_enabled = bool(vad_config.get("enabled", False))
    vad_aggressiveness = int(vad_config.get("aggressiveness", 2))
    vad_frame_ms = int(vad_config.get("frame_ms", 30))
    if vad_frame_ms not in (10, 20, 30):
        vad_frame_ms = 30
    if vad_enabled and webrtcvad is None:
        print("[warning] webrtcvad not installed. VAD disabled.")
        vad_enabled = False

    chunk = int(rate * (vad_frame_ms / 1000.0))
    vad = webrtcvad.Vad(vad_aggressiveness) if vad_enabled else None

    print(f"[system] 录音开始 ({stop_hint})...")

    try:
        # 打开麦克风输入流
        stream_kwargs = {
            "format": audio_format,
            "channels": channels,
            "rate": rate,
            "input": True,  # input=True 即"从麦克风读取"
            "frames_per_buffer": chunk,
        }
        if input_device_index is not None:
            stream_kwargs["input_device_index"] = int(input_device_index)

        stream = audio.open(**stream_kwargs)

        # 把各种信息保存到全局状态字典中
        # 这样 stop_recording() 函数可以访问这些信息
        recording_state["stream"] = stream
        recording_state["frames"] = []              # 用来存放录下来的音频数据块
        recording_state["audio"] = audio
        recording_state["is_recording"] = True      # 标记"正在录音"
        recording_state["chunk"] = chunk
        recording_state["output_path"] = output_path
        recording_state["channels"] = channels
        recording_state["audio_format"] = audio_format
        recording_state["rate"] = rate
        recording_state["input_device_index"] = input_device_index
        recording_state["vad_enabled"] = vad_enabled
        recording_state["vad"] = vad
        recording_state["vad_frame_ms"] = vad_frame_ms
        recording_state["vad_start_time"] = time.time()
        recording_state["vad_last_voice_time"] = recording_state["vad_start_time"]

        # 创建一个"后台线程"来读取音频
        # 这样主程序可以继续响应快捷键（按 Alt+3），不会被卡住
        import threading

        def read_audio():
            """在后台持续读取音频，直到 is_recording 变成 False"""
            while recording_state["is_recording"]:
                try:
                    # 从麦克风读取一块音频数据
                    data = stream.read(chunk, exception_on_overflow=False)
                    if recording_state.get("vad_enabled"):
                        vad_instance = recording_state.get("vad")
                        if vad_instance and vad_instance.is_speech(data, rate):
                            recording_state["vad_last_voice_time"] = time.time()
                    # 把这块数据加到列表里，等待保存
                    recording_state["frames"].append(data)
                except Exception as e:
                    print(f"[error] 读取音频失败: {e}")
                    break

        # 创建并启动线程
        thread = threading.Thread(target=read_audio, daemon=True)
        thread.start()

        return True
    except Exception as exc:
        print(f"[error] 开始录音失败: {exc}")
        audio.terminate()
        return False


def stop_recording():
    """
    停止录音（由主流程热键触发）
    
    原理：
    1. 叫后台线程停止读取
    2. 关闭麦克风
    3. 把所有音频数据写入到 WAV 文件
    
    返回: True（成功） 或 False（失败）
    """
    # 检查有没有进行中的录音
    if not recording_state["is_recording"]:
        print("[system] 还没有开始录音")
        return False

    # 设置标志位，告诉后台线程停止读取
    recording_state["is_recording"] = False

    try:
        # 从全局状态中取出各种信息
        stream = recording_state["stream"]
        audio = recording_state["audio"]
        frames = recording_state["frames"]

        # 停止读取并关闭麦克风
        stream.stop_stream()
        stream.close()
        audio.terminate()

        # 取出录音信息
        output_path = recording_state["output_path"]
        channels = recording_state["channels"]
        audio_format = recording_state["audio_format"]
        rate = recording_state["rate"]

        # 将所有音频数据块写入到 WAV 文件
        with wave.open(str(output_path), "wb") as wav_file:
            wav_file.setnchannels(channels)                                    # 设置声道数
            wav_file.setsampwidth(audio.get_sample_size(audio_format))         # 设置采样位深
            wav_file.setframerate(rate)                                        # 设置采样率
            wav_file.writeframes(b"".join(frames))                             # 写入所有音频数据

        print(f"[system] 录音已保存: {output_path}")
        return True
    except Exception as exc:
        print(f"[error] 停止录音失败: {exc}")
        return False


def record_with_hotkeys(filename=DEFAULT_OUTPUT):
    """
    启动快捷键监听，控制录音的开始和停止
    
    快捷键设置（演示模式）：
    - Alt+1：开始录音
    - Alt+3：停止录音
    - Esc：退出程序
    
    参数: filename - 录音保存文件路径
    """
    print("按 Alt+1 开始录音")
    print("按 Alt+3 停止录音")
    print("按 Esc 退出程序")

    # 绑定快捷键
    # 当按下某个快捷键时，就会调用后面的函数
    keyboard.add_hotkey("alt+1", lambda: start_recording(filename))
    keyboard.add_hotkey("alt+3", stop_recording)

    # keyboard.wait() 会让程序一直运行，直到你按下指定的键（这里是 Esc）
    keyboard.wait("esc")
    print("[system] 程序退出")


# "主程序"入口：当你直接运行这个文件时会执行
if __name__ == "__main__":
    record_with_hotkeys()
