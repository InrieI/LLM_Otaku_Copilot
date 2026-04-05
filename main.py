"""
LLM_OtakuWifu_Copilot 主程序
全局热键驱动的语音助手：录音 -> 转写 -> LLM -> TTS -> 播放
"""

from datetime import datetime
from pathlib import Path
import audioop
import base64
import json
import threading
import time
import wave

import keyboard
import requests

from speech import recorder, transcriber
from llm.client import load_config, generate_chat

try:
    import winsound
except ImportError:
    winsound = None

try:
    import mss
    import mss.tools
except ImportError:
    mss = None

try:
    from PIL import Image, ImageGrab
except ImportError:
    Image = None
    ImageGrab = None


OUTPUT_DIR = Path("outputs")
HISTORY_PATH = OUTPUT_DIR / "chat_history.jsonl"
LAST_REPLY_TEXT_PATH = OUTPUT_DIR / "latest_reply.txt"
LAST_REPLY_AUDIO_PATH = OUTPUT_DIR / "latest_reply.wav"
TTS_CONFIG_PATH = OUTPUT_DIR / "tts_config.json"
SCREEN_CONFIG_PATH = OUTPUT_DIR / "screen_config.json"
SCREENSHOT_DIR = OUTPUT_DIR / "screenshots"

SYSTEM_PROMPT = "你的名字是响，是一个慵懒且理智冷静的15岁少女，你和用户是好朋友关系。你不需要对他使用敬语，可以对他开玩笑互损。说话要简短，要直白，类似正常朋友之间的交流。你不需要刻意模仿少女的说话方式，但要保持自然和真实。总之，你就是一个普通的15岁少女，和用户是好朋友，聊天要自然，不要太严肃。"

DEFAULT_TTS_CONFIG = {
    "base_url": "http://127.0.0.1:9880",
    "text_lang": "zh",
    "ref_audio_path": "",
    "aux_ref_audio_paths": [],
    "prompt_lang": "ja",
    "prompt_text": "",
    "text_split_method": "cut3",
    "batch_size": 1,
    "batch_threshold": 0.75,
    "split_bucket": True,
    "sample_steps": 32,
    "fragment_interval": 0.3,
    "speed_factor": 1.0,
    "top_k": 5,
    "top_p": 1.0,
    "temperature": 1.0,
    "seed": -1,
    "repetition_penalty": 1.35,
    "media_type": "wav",
    "streaming_mode": False,
    "parallel_infer": True,
    "super_sampling": False,
    "overlap_length": 2,
    "min_chunk_length": 16,
    "volume": 0.7,
}

DEFAULT_SCREEN_CONFIG = {
    "monitor_index": 1,
    "max_dim": 1080,
    "jpeg_quality": 70,
}

_processing_lock = threading.Lock()
_pending_screenshot = False


def _ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _ensure_screenshot_dir():
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path, default_value):
    if not path.exists():
        return default_value
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default_value


def _save_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _normalize_tts_config(data):
    config = dict(DEFAULT_TTS_CONFIG)
    if isinstance(data, dict):
        config.update({k: v for k, v in data.items() if k in config})

    aux = config.get("aux_ref_audio_paths", [])
    if isinstance(aux, str):
        aux = [aux]
    if not isinstance(aux, list):
        aux = []
    config["aux_ref_audio_paths"] = aux

    return config


def _normalize_screen_config(data):
    config = dict(DEFAULT_SCREEN_CONFIG)
    if isinstance(data, dict):
        config.update({k: v for k, v in data.items() if k in config})
    try:
        config["monitor_index"] = int(config.get("monitor_index", 1))
    except (TypeError, ValueError):
        config["monitor_index"] = 1
    if config["monitor_index"] < 1:
        config["monitor_index"] = 1
    config["max_dim"] = _clamp_max_dim(config.get("max_dim", 1080))
    config["jpeg_quality"] = _clamp_jpeg_quality(config.get("jpeg_quality", 70))
    return config


def _prompt_if_missing(config, key, label):
    value = str(config.get(key, "")).strip()
    if value:
        return config
    print(f"[setup] {label} 为空，请输入。")
    config[key] = input("> ").strip()
    return config


def _format_aux_list(aux_list):
    if not aux_list:
        return "(无)"
    return "\n".join(f"  {idx + 1}) {path}" for idx, path in enumerate(aux_list))


def _prompt_with_default(config, key, label, hint=None):
    default_value = config.get(key, "")
    if hint:
        print(f"[setup] {label}（{hint}）")
    else:
        print(f"[setup] {label}")
    value = input(f"(回车使用默认: {default_value})\n> ").strip()
    if value:
        config[key] = value
    return config


def _prompt_monitor_index(config):
    if mss is None:
        print("[warning] 未安装 mss，无法列出多屏，默认截图为全屏。")
        return config

    with mss.mss() as sct:
        monitors = sct.monitors[1:]

    if not monitors:
        print("[warning] 未检测到屏幕信息，默认截图为全屏。")
        return config

    default_index = config.get("monitor_index", 1)
    print("[setup] 可用屏幕：")
    for idx, mon in enumerate(monitors, start=1):
        print(f"  {idx}) {mon['width']}x{mon['height']} ({mon['left']},{mon['top']})")

    value = input(f"选择默认截图屏幕 (回车使用默认: {default_index})\n> ").strip()
    if value.isdigit():
        choice = int(value)
        if 1 <= choice <= len(monitors):
            config["monitor_index"] = choice
    return config


def _clamp_max_dim(value):
    try:
        dim = int(value)
    except (TypeError, ValueError):
        return 1080
    if dim < 320:
        return 320
    return dim


def _clamp_jpeg_quality(value):
    try:
        quality = int(value)
    except (TypeError, ValueError):
        return 70
    if quality < 50:
        return 50
    if quality > 75:
        return 75
    return quality


def _prompt_number(config, key, label, is_int=False):
    default_value = config.get(key, "")
    value = input(f"[setup] {label} (回车使用默认: {default_value})\n> ").strip()
    if not value:
        return config
    try:
        config[key] = int(value) if is_int else float(value)
    except ValueError:
        print("[warning] 输入不是数字，已保持默认值")
    return config


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


def _normalize_path(value):
    if not isinstance(value, str):
        return ""
    cleaned = value.strip().strip('"').strip("'")
    return cleaned.replace("\\", "/")


def _prompt_aux_refs(config):
    if config.get("aux_ref_audio_paths"):
        return config
    print("[setup] 副参考音频路径(可选，多个用逗号分隔，直接回车跳过)：")
    raw = input("> ").strip()
    if not raw:
        return config
    parts = [p.strip() for p in raw.replace(";", ",").split(",")]
    paths = [_normalize_path(p) for p in parts if p.strip()]
    config["aux_ref_audio_paths"] = paths
    return config


def _prompt_ref_audio_update(config):
    current = _normalize_path(config.get("ref_audio_path", ""))
    if current:
        print(f"[setup] 当前主参考音频路径: {current}")
        value = input("回车保留当前，输入新路径修改\n> ").strip()
        if value:
            config["ref_audio_path"] = value
    return config


def _prompt_prompt_text_update(config):
    current = str(config.get("prompt_text", "")).strip()
    if current:
        print(f"[setup] 当前参考音频台词: {current}")
        value = input("回车保留当前，输入新台词修改\n> ").strip()
        if value:
            config["prompt_text"] = value
    return config


def _prompt_aux_refs_update(config):
    current = config.get("aux_ref_audio_paths", [])
    print("[setup] 当前副参考音频路径列表:")
    print(_format_aux_list(current))
    value = input("回车保留当前，输入 clear 清空，或输入新路径(逗号分隔)\n> ").strip()
    if value.lower() == "clear":
        config["aux_ref_audio_paths"] = []
    elif value:
        parts = [p.strip() for p in value.replace(";", ",").split(",")]
        paths = [_normalize_path(p) for p in parts if p.strip()]
        config["aux_ref_audio_paths"] = paths
    return config


def _should_update_tts_config(has_existing_file):
    if not has_existing_file:
        return True
    answer = input("[setup] 是否更新 TTS 高级参数？(y/N)\n> ").strip().lower()
    return answer == "y"


def load_tts_config():
    _ensure_output_dir()
    has_existing_file = TTS_CONFIG_PATH.exists()
    raw = _load_json(TTS_CONFIG_PATH, DEFAULT_TTS_CONFIG)
    config = _normalize_tts_config(raw)

    config = _prompt_ref_audio_update(config)
    config = _prompt_prompt_text_update(config)
    config = _prompt_if_missing(config, "ref_audio_path", "主参考音频路径")
    config = _prompt_if_missing(config, "prompt_text", "参考音频台词")
    config = _prompt_if_missing(config, "prompt_lang", "参考音频语言(例: ja)")
    config = _prompt_if_missing(
        config,
        "text_lang",
        "目标文本语言(可用: auto/auto_yue/en/zh/ja/yue/ko/all_zh/all_ja/all_yue/all_ko; zh=中英混合)",
    )
    config = _prompt_aux_refs_update(config)

    if _should_update_tts_config(has_existing_file):
        config = _prompt_with_default(
            config,
            "text_lang",
            "目标文本语言",
            "auto/auto_yue/en/zh/ja/yue/ko/all_zh/all_ja/all_yue/all_ko; zh=中英混合",
        )
        config = _prompt_with_default(config, "prompt_lang", "参考音频语言")
        config = _prompt_with_default(
            config,
            "text_split_method",
            "切分方式",
            "cut3=中文句号 | cut5=按标点",
        )
        config = _prompt_number(config, "speed_factor", "语速")
        config = _prompt_number(config, "top_k", "top_k", is_int=True)
        config = _prompt_number(config, "top_p", "top_p")
        config = _prompt_number(config, "temperature", "temperature")
        config = _prompt_number(config, "repetition_penalty", "重复惩罚")
        config = _prompt_number(config, "batch_size", "batch_size", is_int=True)
        config = _prompt_number(config, "sample_steps", "采样步数", is_int=True)
        config = _prompt_number(config, "fragment_interval", "分段间隔(秒)")
        config = _prompt_number(config, "volume", "播放音量(0.0-1.0)")

    config["ref_audio_path"] = _normalize_path(config.get("ref_audio_path", ""))
    config["aux_ref_audio_paths"] = [
        _normalize_path(path) for path in config.get("aux_ref_audio_paths", [])
    ]
    config["volume"] = _clamp_volume(config.get("volume", 1.0))

    _save_json(TTS_CONFIG_PATH, config)
    return config


def load_screen_config():
    _ensure_output_dir()
    has_existing_file = SCREEN_CONFIG_PATH.exists()
    raw = _load_json(SCREEN_CONFIG_PATH, DEFAULT_SCREEN_CONFIG)
    config = _normalize_screen_config(raw)

    if not has_existing_file:
        config = _prompt_monitor_index(config)
        _save_json(SCREEN_CONFIG_PATH, config)
    return config


def append_history(role, content, meta=None):
    if not content:
        return
    _ensure_output_dir()
    record = {
        "role": role,
        "content": content,
        "time": datetime.utcnow().isoformat() + "Z",
    }
    if meta:
        record["meta"] = meta
    with HISTORY_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_history():
    if not HISTORY_PATH.exists():
        return []
    history = []
    for line in HISTORY_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        role = item.get("role")
        content = item.get("content")
        if role and content:
            history.append({"role": role, "content": content})
    return history


def clear_history():
    if HISTORY_PATH.exists():
        HISTORY_PATH.write_text("", encoding="utf-8")
    if SCREENSHOT_DIR.exists():
        for item in SCREENSHOT_DIR.glob("*.*"):
            try:
                item.unlink()
            except OSError:
                print(f"[warning] 无法删除截图: {item}")
    print("[system] New session started.")


def build_messages(history):
    messages = []
    if SYSTEM_PROMPT:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.extend(history)
    return messages


def _decode_base64(text):
    try:
        return base64.b64decode(text, validate=True)
    except Exception:
        return None


def _save_audio_from_response(response, output_path: Path):
    content_type = response.headers.get("content-type", "").lower()
    if content_type.startswith("audio/") or content_type == "application/octet-stream":
        output_path.write_bytes(response.content)
        return output_path

    raw_text = response.text.strip()
    if raw_text:
        candidate = Path(raw_text.strip('"'))
        if candidate.exists():
            output_path.write_bytes(candidate.read_bytes())
            return output_path

    try:
        data = response.json()
    except Exception:
        data = None

    if isinstance(data, str):
        candidate = Path(data)
        if candidate.exists():
            output_path.write_bytes(candidate.read_bytes())
            return output_path
        decoded = _decode_base64(data)
        if decoded:
            output_path.write_bytes(decoded)
            return output_path

    if isinstance(data, dict):
        for key in ("path", "file", "audio_path", "audio", "data", "wav", "content"):
            value = data.get(key)
            if isinstance(value, str):
                candidate = Path(value)
                if candidate.exists():
                    output_path.write_bytes(candidate.read_bytes())
                    return output_path
                decoded = _decode_base64(value)
                if decoded:
                    output_path.write_bytes(decoded)
                    return output_path

    return None


def _resize_image(image, max_dim):
    if image is None:
        return None
    width, height = image.size
    scale = max(width, height) / float(max_dim)
    if scale <= 1.0:
        return image
    new_size = (int(width / scale), int(height / scale))
    return image.resize(new_size, Image.LANCZOS)


def _capture_screenshot(screen_config):
    _ensure_output_dir()
    _ensure_screenshot_dir()
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_path = SCREENSHOT_DIR / f"screen_{timestamp}.jpg"
    max_dim = _clamp_max_dim(screen_config.get("max_dim", 1080))
    quality = _clamp_jpeg_quality(screen_config.get("jpeg_quality", 70))

    image = None

    if mss is not None:
        with mss.mss() as sct:
            monitors = sct.monitors
            index = int(screen_config.get("monitor_index", 1))
            if index < 1 or index >= len(monitors):
                index = 1
            shot = sct.grab(monitors[index])
            if Image is not None:
                image = Image.frombytes("RGB", shot.size, shot.rgb)
            else:
                png_path = SCREENSHOT_DIR / f"screen_{timestamp}.png"
                mss.tools.to_png(shot.rgb, shot.size, output=str(png_path))
                print("[warning] 未安装 Pillow，已保存 PNG 原图。")
                return png_path

    if image is None and ImageGrab is not None:
        image = ImageGrab.grab()

    if image is None:
        print("[warning] 缺少截图依赖，请安装 mss 或 Pillow。")
        return None

    if Image is None:
        print("[warning] 未安装 Pillow，无法压缩截图。")
        return None

    image = _resize_image(image, max_dim)
    image = image.convert("RGB")
    image.save(output_path, "JPEG", quality=quality, optimize=True)
    return output_path


def _image_to_data_url(path: Path):
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    ext = path.suffix.lower()
    mime = "image/jpeg" if ext in {".jpg", ".jpeg"} else "image/png"
    return f"data:{mime};base64,{encoded}"


def _attach_image_to_messages(messages, image_path: Path):
    if not image_path or not image_path.exists():
        return messages
    data_url = _image_to_data_url(image_path)
    for msg in reversed(messages):
        if msg.get("role") == "user":
            text = msg.get("content")
            if not isinstance(text, str):
                text = ""
            msg["content"] = [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]
            return messages
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": ""},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    )
    return messages


def synthesize_tts(text, tts_config):
    base_url = tts_config.get("base_url", "http://127.0.0.1:9880").rstrip("/")
    url = f"{base_url}/tts"

    payload = dict(tts_config)
    payload["text"] = text
    payload.pop("base_url", None)

    response = requests.post(url, json=payload, timeout=120)
    if response.status_code != 200:
        print(f"[error] TTS failed: {response.status_code} {response.text}")
        return None

    _ensure_output_dir()
    saved = _save_audio_from_response(response, LAST_REPLY_AUDIO_PATH)
    if not saved:
        print("[error] Failed to parse TTS response.")
    return saved


def play_audio(path: Path):
    if not path or not path.exists():
        return
    if winsound is None:
        print(f"[warning] Audio saved: {path}")
        return
    winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)


def _apply_volume_wav(path: Path, volume):
    volume = _clamp_volume(volume)
    if volume >= 0.999:
        return path

    scaled_path = OUTPUT_DIR / "latest_reply_scaled.wav"
    try:
        with wave.open(str(path), "rb") as reader:
            params = reader.getparams()
            frames = reader.readframes(reader.getnframes())
        scaled_frames = audioop.mul(frames, params.sampwidth, volume)
        with wave.open(str(scaled_path), "wb") as writer:
            writer.setparams(params)
            writer.writeframes(scaled_frames)
        return scaled_path
    except Exception:
        print("[warning] Failed to apply volume, playing original audio.")
        return path


def _process_pipeline(llm_config, tts_config, with_screenshot=False):
    if not recorder.stop_recording():
        return

    text = transcriber.transcribe_audio()
    if not text or not text.strip():
        print("[system] 转录为空，已忽略。")
        return

    screenshot_path = None
    if with_screenshot:
        screen_config = load_screen_config()
        screenshot_path = _capture_screenshot(screen_config)

    meta = None
    if screenshot_path:
        meta = {"screenshot": screenshot_path.as_posix()}

    append_history("user", text, meta=meta)
    history = load_history()
    messages = build_messages(history)

    if screenshot_path:
        provider = llm_config.provider.lower()
        if provider in {"openai", "openai_compat", "api"}:
            messages = _attach_image_to_messages(messages, screenshot_path)
        else:
            print("[warning] 当前提供商不支持图片，已忽略截图。")

    reply = generate_chat(messages, llm_config)
    if not reply:
        return

    append_history("assistant", reply)
    _ensure_output_dir()
    LAST_REPLY_TEXT_PATH.write_text(reply, encoding="utf-8")

    audio_path = synthesize_tts(reply, tts_config)
    if audio_path:
        adjusted_path = _apply_volume_wav(audio_path, tts_config.get("volume", 1.0))
        play_audio(adjusted_path)


def handle_stop_hotkey(llm_config, tts_config):
    if not _processing_lock.acquire(blocking=False):
        print("[system] Busy. Please wait...")
        return

    global _pending_screenshot
    with_screenshot = _pending_screenshot
    _pending_screenshot = False

    def _runner():
        try:
            _process_pipeline(llm_config, tts_config, with_screenshot=with_screenshot)
        finally:
            _processing_lock.release()

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()


def handle_start_hotkey(with_screenshot=False):
    if _processing_lock.locked():
        print("[system] 正在处理上一段，请稍后再录音。")
        return
    if recorder.recording_state.get("is_recording"):
        print("[system] 已在录音中。")
        return
    global _pending_screenshot
    _pending_screenshot = with_screenshot
    recorder.start_recording()


def main():
    llm_config = load_config(interactive=True)
    tts_config = load_tts_config()

    print("[system] Assistant ready.")
    print("Hotkeys:")
    print("  Alt+1 -> start recording")
    print("  Alt+2 -> start recording (with screenshot)")
    print("  Alt+3 -> stop and process")
    print("  Alt+0 -> clear history")
    print("  Esc   -> exit")

    keyboard.add_hotkey("alt+1", lambda: handle_start_hotkey(with_screenshot=False))
    keyboard.add_hotkey("alt+2", lambda: handle_start_hotkey(with_screenshot=True))
    keyboard.add_hotkey("alt+3", lambda: handle_stop_hotkey(llm_config, tts_config))
    keyboard.add_hotkey("alt+0", clear_history)

    keyboard.wait("esc")
    print("[system] Exiting.")


if __name__ == "__main__":
    main()
