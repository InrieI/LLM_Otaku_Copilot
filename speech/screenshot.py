from datetime import datetime, timezone
from pathlib import Path
import base64
import json

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


DEFAULT_SCREEN_CONFIG = {
    "monitor_index": 1,
    "max_dim": 1080,
    "jpeg_quality": 70,
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


def load_screen_config(output_dir=Path("outputs")):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "screen_config.json"

    has_existing_file = config_path.exists()
    raw = _load_json(config_path, DEFAULT_SCREEN_CONFIG)
    config = _normalize_screen_config(raw)

    if not has_existing_file:
        config = _prompt_monitor_index(config)
        _save_json(config_path, config)

    return config


def _resize_image(image, max_dim):
    width, height = image.size
    scale = max(width, height) / float(max_dim)
    if scale <= 1.0:
        return image
    new_size = (int(width / scale), int(height / scale))
    return image.resize(new_size, Image.LANCZOS)


def capture_screenshot(screen_config, output_dir=Path("outputs")):
    output_dir = Path(output_dir)
    screenshot_dir = output_dir / "screenshots"
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = screenshot_dir / f"screen_{timestamp}.jpg"
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
                png_path = screenshot_dir / f"screen_{timestamp}.png"
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


def image_to_data_url(path):
    path = Path(path)
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    ext = path.suffix.lower()
    mime = "image/jpeg" if ext in {".jpg", ".jpeg"} else "image/png"
    return f"data:{mime};base64,{encoded}"


def attach_image_to_messages(messages, image_path):
    image_path = Path(image_path)
    if not image_path.exists():
        return messages

    data_url = image_to_data_url(image_path)

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
