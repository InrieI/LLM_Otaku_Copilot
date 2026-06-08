import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import webview
import threading
import json
import socket
from pathlib import Path
from pydantic import BaseModel
import sys
import os
import time

from workflows import VoiceChatWorkflow
from speech import recorder

try:
    import keyboard
except Exception:
    keyboard = None

try:
    import pyaudio
except Exception:
    pyaudio = None

try:
    from speech.wake_word import WakeWordListener
except Exception:
    WakeWordListener = None

try:
    from speech.custom_wakeword import CustomWakeWordListener, train_custom_wakeword, dataset_stats
except Exception:
    CustomWakeWordListener = None
    train_custom_wakeword = None
    dataset_stats = None

app = FastAPI()
frontend_dir = Path(__file__).resolve().parent / "frontend"
config_path = Path(__file__).resolve().parent / "config.json"
workflow = VoiceChatWorkflow()
wakeword_listener = None
outputs_dir = Path(__file__).resolve().parent / "outputs"
wakeword_dataset_dir = outputs_dir / "wakeword_dataset"
wakeword_models_dir = outputs_dir / "wakeword_models"
wakeword_training_state = {"status": "idle", "message": "", "name": ""}
wakeword_sample_state = {"path": None, "label": None, "name": None}
live2d_models_dir = Path(__file__).resolve().parent / "live2d-models"
characters_dir = Path(__file__).resolve().parent / "characters"
live2d_state = {"emotion": "neutral", "reply_text": "", "audio_version": 0}
workflow.live2d_state = live2d_state
hotkey_handles = []

# Serve static files for frontend
app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

# Serve Live2D model files
if live2d_models_dir.exists():
    app.mount("/live2d-models", StaticFiles(directory=str(live2d_models_dir)), name="live2d-models")

# Serve character portrait files
if characters_dir.exists():
    app.mount("/characters", StaticFiles(directory=str(characters_dir)), name="characters")

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open(frontend_dir / "index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/live2d", response_class=HTMLResponse)
async def get_live2d():
    live2d_html = frontend_dir / "live2d.html"
    if not live2d_html.exists():
        return HTMLResponse("<h1>live2d.html not found</h1>", status_code=404)
    with open(live2d_html, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/galgame", response_class=HTMLResponse)
async def get_galgame():
    galgame_html = frontend_dir / "galgame.html"
    if not galgame_html.exists():
        return HTMLResponse("<h1>galgame.html not found</h1>", status_code=404)
    with open(galgame_html, "r", encoding="utf-8") as f:
        return f.read()


def _scan_characters():
    """Scan characters/ for subdirectories containing emotion images.

    Supports single images (joy.png) and numbered variants (joy1.png, joy2.png, ...).
    Returns {name, images: {emotion: [url, ...]}}.
    """
    import re as _re
    result = []
    if not characters_dir.exists():
        return result
    known_emotions = {"neutral", "joy", "sadness", "anger", "surprise", "fear", "disgust", "smirk"}
    for child in sorted(characters_dir.iterdir()):
        if not child.is_dir():
            continue
        images = {}
        for f in child.iterdir():
            if not f.is_file():
                continue
            ext = f.suffix.lower().lstrip(".")
            if ext not in ("png", "jpg", "jpeg", "webp", "avif", "gif"):
                continue
            stem = f.stem  # e.g. "joy", "joy1", "joy2"
            # Match emotion name + optional number suffix
            m = _re.match(r'^([a-z]+)(\d*)$', stem)
            if not m:
                continue
            emo = m.group(1)
            if emo not in known_emotions:
                continue
            url = f"/characters/{child.name}/{f.name}"
            images.setdefault(emo, []).append(url)
        if images:
            # Sort each emotion's variants for stable order
            for emo in images:
                images[emo].sort()
            result.append({"name": child.name, "images": images})
    return result


@app.get("/api/characters")
async def list_characters():
    return {"characters": _scan_characters()}


def _scan_live2d_models():
    """Scan live2d-models/ for .model3.json files and return model configs."""
    models = []
    if not live2d_models_dir.exists():
        return models
    for child in sorted(live2d_models_dir.iterdir()):
        if not child.is_dir():
            continue
        # Search for .model3.json recursively (may be in subfolders like runtime/)
        model3_files = list(child.rglob("*.model3.json"))
        if not model3_files:
            continue
        model3_path = model3_files[0]
        rel_path = model3_path.relative_to(live2d_models_dir)
        url = f"/live2d-models/{rel_path.as_posix()}"
        models.append({
            "name": child.name,
            "url": url,
            "kScale": 0.3,
            "initialXshift": 0,
            "initialYshift": 0,
            "idleMotionGroupName": "Idle",
            "emotionMap": {
                "neutral": 0, "joy": 3, "sadness": 1,
                "anger": 2, "surprise": 3, "fear": 1,
                "disgust": 2, "smirk": 3
            },
        })
    return models


@app.get("/api/live2d/models")
async def live2d_models():
    models = _scan_live2d_models()
    # Check config for selected model and overrides
    selected_name = None
    overrides = {}
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            l2d_cfg = cfg.get("live2d", {})
            selected_name = l2d_cfg.get("model_name")
            overrides = l2d_cfg.get("model_overrides", {})
        except Exception:
            pass
    # Apply overrides and find current
    current = None
    for m in models:
        if m["name"] in overrides:
            m.update(overrides[m["name"]])
        if m["name"] == selected_name:
            current = m
    if current is None and models:
        current = models[0]
    return {"models": models, "current": current}


@app.get("/api/live2d/state")
async def live2d_get_state():
    return live2d_state


@app.get("/api/live2d/audio")
async def live2d_audio():
    audio_path = outputs_dir / "latest_reply_scaled.wav"
    if not audio_path.exists():
        audio_path = outputs_dir / "latest_reply.wav"
    if not audio_path.exists():
        return {"error": "no audio"}
    return FileResponse(
        path=str(audio_path),
        media_type="audio/wav",
        headers={"Cache-Control": "no-cache"},
    )

@app.get("/api/config")
async def get_config():
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

@app.post("/api/config")
async def save_config(request: Request):
    data = await request.json()
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    workflow.reload_config()
    wakeword_cfg = _load_wakeword_config()
    _restart_wakeword_listener(wakeword_cfg)
    return {"status": "success"}


class RecordToggleRequest(BaseModel):
    with_screenshot: bool = False


class WakewordSampleRequest(BaseModel):
    name: str
    label: str


class WakewordTrainRequest(BaseModel):
    name: str
    window_seconds: float = 1.5


def _record_status():
    return {
        "is_recording": bool(recorder.recording_state.get("is_recording")),
        "pending_screenshot": bool(workflow.pending_screenshot),
    }


def _get_recording_device_index():
    if not config_path.exists():
        return None
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        recording_cfg = data.get("recording", {})
        if not isinstance(recording_cfg, dict):
            return None
        device_index = recording_cfg.get("device_index")
        return device_index if isinstance(device_index, int) else None
    except Exception:
        return None


@app.get("/api/record/status")
async def record_status():
    return _record_status()


@app.post("/api/record/toggle")
async def record_toggle(payload: RecordToggleRequest):
    workflow.start_or_stop_recording(with_screenshot=payload.with_screenshot)
    return _record_status()


@app.post("/api/record/cancel")
async def record_cancel():
    workflow.force_stop_only()
    return _record_status()


@app.post("/api/record/clear")
async def record_clear():
    workflow.clear_history()
    return {"status": "success"}


def _list_input_devices():
    if pyaudio is None:
        return []
    audio = pyaudio.PyAudio()
    devices = []
    try:
        default_info = None
        try:
            default_info = audio.get_default_input_device_info()
        except Exception:
            default_info = None

        for idx in range(audio.get_device_count()):
            info = audio.get_device_info_by_index(idx)
            if int(info.get("maxInputChannels", 0)) <= 0:
                continue
            devices.append(
                {
                    "index": int(info.get("index")),
                    "name": str(info.get("name", "")),
                    "channels": int(info.get("maxInputChannels", 1)),
                    "defaultSampleRate": int(info.get("defaultSampleRate", 0)),
                    "isDefault": bool(default_info and info.get("index") == default_info.get("index")),
                }
            )
    finally:
        audio.terminate()
    return devices


@app.get("/api/audio/devices")
async def audio_devices():
    return {"devices": _list_input_devices()}


def _normalize_wakeword_name(name: str) -> str:
    cleaned = "".join(c for c in name.lower().strip() if c.isalnum() or c in "-_ ")
    cleaned = cleaned.replace(" ", "_")
    return cleaned or "custom_wakeword"


@app.get("/api/wakeword/train/status")
async def wakeword_train_status():
    return wakeword_training_state


@app.get("/api/wakeword/samples/stats")
async def wakeword_sample_stats(name: str):
    if dataset_stats is None:
        return {"positive": 0, "negative": 0}
    dataset_dir = wakeword_dataset_dir / _normalize_wakeword_name(name)
    return dataset_stats(dataset_dir)


@app.get("/api/wakeword/custom_models")
async def get_custom_models():
    models = []
    if wakeword_models_dir.exists():
        for p in wakeword_models_dir.glob("*.json"):
            models.append(p.stem)
    return {"models": sorted(models)}


@app.get("/api/wakeword/samples/list")
async def wakeword_sample_list(name: str):
    name = _normalize_wakeword_name(name)
    dataset_dir = wakeword_dataset_dir / name
    pos_files = []
    neg_files = []
    if (dataset_dir / "positive").exists():
        pos_files = [p.name for p in (dataset_dir / "positive").glob("*.wav")]
    if (dataset_dir / "negative").exists():
        neg_files = [p.name for p in (dataset_dir / "negative").glob("*.wav")]
    return {"positive": sorted(pos_files, reverse=True), "negative": sorted(neg_files, reverse=True)}


class WakewordSampleDeleteRequest(BaseModel):
    name: str
    label: str
    filename: str


@app.post("/api/wakeword/samples/delete")
async def wakeword_sample_delete(payload: WakewordSampleDeleteRequest):
    name = _normalize_wakeword_name(payload.name)
    label = payload.label.lower().strip()
    if label not in {"positive", "negative"}:
         return {"status": "error", "message": "无效的样本标签"}
    target_file = wakeword_dataset_dir / name / label / payload.filename
    if target_file.exists():
        try:
            target_file.unlink()
        except Exception as exc:
            return {"status": "error", "message": str(exc)}
    return {"status": "success"}


@app.post("/api/wakeword/samples/start")
async def wakeword_sample_start(payload: WakewordSampleRequest):
    if recorder.recording_state.get("is_recording"):
        return {"status": "error", "message": "已有录音进行中"}

    name = _normalize_wakeword_name(payload.name)
    label = payload.label.lower().strip()
    if label not in {"positive", "negative"}:
        return {"status": "error", "message": "无效的样本标签"}

    target_dir = wakeword_dataset_dir / name / label
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = target_dir / f"{int(time.time())}.wav"

    input_device_index = _get_recording_device_index()
    recorder.start_recording(
        filename=filename,
        stop_hint="再次点击停止采样",
        vad_config={"enabled": False},
        input_device_index=input_device_index,
    )
    wakeword_sample_state.update({"path": str(filename), "label": label, "name": name})
    return {"status": "recording", "path": str(filename)}


@app.post("/api/wakeword/samples/stop")
async def wakeword_sample_stop():
    if not recorder.recording_state.get("is_recording"):
        return {"status": "idle"}
    recorder.stop_recording()
    return {"status": "stopped", "path": wakeword_sample_state.get("path")}


@app.post("/api/wakeword/train")
async def wakeword_train(payload: WakewordTrainRequest):
    if train_custom_wakeword is None:
        return {"status": "error", "message": "训练组件不可用"}
    if wakeword_training_state.get("status") == "running":
        return {"status": "running", "message": "训练正在进行中"}

    name = _normalize_wakeword_name(payload.name)
    dataset_dir = wakeword_dataset_dir / name
    model_path = wakeword_models_dir / f"{name}.json"

    def _runner():
        wakeword_training_state.update({"status": "running", "message": "", "name": name})
        try:
            train_custom_wakeword(
                dataset_dir=dataset_dir,
                output_path=model_path,
                window_seconds=payload.window_seconds,
                prefer_gpu=True,
            )

            current = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
            wakeword_cfg = current.get("wakeword", {}) if isinstance(current, dict) else {}
            wakeword_cfg.update({
                "mode": "custom",
                "model": str(model_path),
                "enabled": True,
            })
            current["wakeword"] = wakeword_cfg
            config_path.write_text(json.dumps(current, indent=2, ensure_ascii=False), encoding="utf-8")
            workflow.reload_config()
            _restart_wakeword_listener(wakeword_cfg)

            wakeword_training_state.update({"status": "done", "message": "训练完成"})
        except Exception as exc:
            wakeword_training_state.update({"status": "error", "message": str(exc)})

    threading.Thread(target=_runner, daemon=True).start()
    return {"status": "running"}

def _get_all_lan_ips():
    """Get all available LAN IPs with adapter names, sorted by likelihood of being WiFi/Ethernet."""
    results = []
    seen = set()

    # Method 1: Parse ipconfig for adapter names
    adapter_ips = {}
    try:
        import subprocess
        output = subprocess.check_output("ipconfig /all", shell=True, text=True, encoding="mbcs", errors="replace")
        current_adapter = ""
        for line in output.splitlines():
            line = line.strip()
            if line and not line.startswith(" ") and ("adapter" in line.lower() or "适配器" in line):
                current_adapter = line
            elif "IPv4" in line or "IPv4 地址" in line:
                parts = line.split(":")
                if len(parts) >= 2:
                    ip = parts[-1].strip().split("(")[0].strip()
                    if ip and ip not in seen:
                        seen.add(ip)
                        name = current_adapter.lower()
                        # Prioritize: WiFi > Ethernet > others > VPN/virtual
                        priority = 3
                        if any(k in name for k in ("wi-fi", "wifi", "wlan", "无线")):
                            priority = 0
                        elif any(k in name for k in ("ethernet", "以太网")):
                            priority = 1
                        elif any(k in name for k in ("vpn", "vmware", "virtualbox", "hyper-v", "vethernet", "loopback")):
                            priority = 5
                        results.append({"ip": ip, "adapter": current_adapter, "priority": priority})
    except Exception:
        pass

    # Method 2: socket fallback for any IPs missed
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip not in seen and not ip.startswith("127."):
                seen.add(ip)
                results.append({"ip": ip, "adapter": "unknown", "priority": 4})
    except Exception:
        pass

    results.sort(key=lambda x: x["priority"])
    return results


def _get_lan_ip():
    """Get the best LAN IP (WiFi preferred)."""
    ips = _get_all_lan_ips()
    for item in ips:
        ip = item["ip"]
        if ip.startswith(("192.168.", "10.", "172.")):
            second = int(ip.split(".")[1]) if ip.startswith("172.") else 0
            if not ip.startswith("172.") or 16 <= second <= 31:
                return ip
    return "127.0.0.1"


@app.get("/api/network/info")
async def network_info():
    ip = _get_lan_ip()
    all_ips = _get_all_lan_ips()
    return {"ip": ip, "port": 8000, "all_ips": all_ips}


def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")

def _get_current_screenshot_mode():
    """Read the current recording mode from config to respect the UI toggle."""
    if not config_path.exists():
        return False
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        return bool(data.get("recording", {}).get("with_screenshot", False))
    except Exception:
        return False


def _enable_hotkeys():
    global hotkey_handles
    if keyboard is None:
        print("[warning] keyboard 模块未安装，热键功能不可用")
        return
    if hotkey_handles:
        return
    # Both Alt+1 and Alt+2 respect the UI mode setting.
    hotkey_handles = [
        keyboard.add_hotkey("alt+1", lambda: workflow.start_or_stop_recording(with_screenshot=_get_current_screenshot_mode())),
        keyboard.add_hotkey("alt+2", lambda: workflow.start_or_stop_recording(with_screenshot=_get_current_screenshot_mode())),
        keyboard.add_hotkey("alt+3", workflow.force_stop_only),
        keyboard.add_hotkey("alt+0", workflow.clear_history),
    ]
    print("[system] Assistant ready. Hotkeys active in background.")


def _disable_hotkeys():
    global hotkey_handles
    if keyboard is None or not hotkey_handles:
        return
    for handle in hotkey_handles:
        try:
            keyboard.remove_hotkey(handle)
        except Exception:
            pass
    hotkey_handles = []
    print("[system] Assistant ready. Hotkeys disabled (wakeword enabled).")


def run_assistant(enable_hotkeys=True):
    if enable_hotkeys:
        _enable_hotkeys()
    else:
        _disable_hotkeys()
    # We do not block here with wait("esc") because pywebview will block the main thread.


def _load_wakeword_config():
    if not config_path.exists():
        return {}
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        return data.get("wakeword", {}) if isinstance(data, dict) else {}
    except Exception as exc:
        print(f"[warning] Failed to read wakeword config: {exc}")
        return {}


def _get_builtin_wakeword_models():
    try:
        import openwakeword

        return sorted(list(openwakeword.MODELS.keys()))
    except Exception:
        return []


@app.get("/api/wakeword/models")
async def wakeword_models():
    return {"models": _get_builtin_wakeword_models()}


def _start_wakeword_listener(wakeword_cfg):
    global wakeword_listener
    _stop_wakeword_listener()

    if not wakeword_cfg.get("enabled"):
        return False

    # 获取用户在 UI 上选择的麦克风设备索引
    device_index = None
    if config_path.exists():
        try:
            full_cfg = json.loads(config_path.read_text(encoding="utf-8"))
            device_index = full_cfg.get("recording", {}).get("device_index")
        except Exception:
            pass

    cooldown_seconds = float(wakeword_cfg.get("cooldown_seconds", 2.5))
    # Use recording config's with_screenshot (controlled by UI toggle), not wakeword config
    with_screenshot = False
    try:
        full_cfg = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        with_screenshot = bool(full_cfg.get("recording", {}).get("with_screenshot", False))
    except Exception:
        pass

    def _on_wake():
        if recorder.recording_state.get("is_recording"):
            return
        workflow.start_or_stop_recording(with_screenshot=with_screenshot)

    mode = str(wakeword_cfg.get("mode", "openwakeword")).lower()

    # ── 自定义唤醒词模式 ──
    if mode == "custom":
        if CustomWakeWordListener is None:
            print("[warning] Custom wakeword listener unavailable.")
            return False

        # 关键：自定义模式使用 custom_name 字段，而不是 model 字段
        custom_name = str(wakeword_cfg.get("custom_name", "shizuku")).strip()
        if not custom_name:
            print("[warning] Custom wakeword name is empty.")
            return False

        model_path = Path("outputs/wakeword_models") / f"{custom_name}.json"
        if not model_path.exists():
            print(f"[warning] Custom wakeword model not found: {model_path}")
            return False

        threshold = wakeword_cfg.get("threshold")

        try:
            wakeword_listener = CustomWakeWordListener(
                model_path=str(model_path),
                threshold=threshold,
                cooldown_seconds=cooldown_seconds,
                on_wake=_on_wake,
                device_index=device_index,
            )
            wakeword_listener.start()
            return True
        except Exception as exc:
            print(f"[warning] Custom wakeword listener failed: {exc}")
            wakeword_listener = None
            return False

    # ── 内置 openwakeword 模式 ──
    if WakeWordListener is None:
        print("[warning] openwakeword not installed. Wakeword disabled.")
        return False

    model_name = str(wakeword_cfg.get("model", "hey_jarvis")).strip()
    model_path = Path(model_name)
    if not model_path.exists():
        normalized = model_name.lower().replace(" ", "_").replace("-", "_")
        if normalized == "heygoogle":
            normalized = "hey_google"
        model_name = normalized
        builtin_models = _get_builtin_wakeword_models()
        if model_name not in builtin_models:
            if builtin_models:
                print(
                    f"[warning] Wakeword model '{model_name}' not found. Falling back to '{builtin_models[0]}'."
                )
                model_name = builtin_models[0]
            else:
                print("[warning] No built-in wakeword models available.")
                return False
    threshold = float(wakeword_cfg.get("threshold", 0.6))

    try:
        wakeword_listener = WakeWordListener(
            model_name=model_name,
            threshold=threshold,
            cooldown_seconds=cooldown_seconds,
            on_wake=_on_wake,
            device_index=device_index,
        )
        wakeword_listener.start()
        return True
    except Exception as exc:
        print(f"[warning] Wakeword listener failed: {exc}")
        wakeword_listener = None
        return False


def _stop_wakeword_listener():
    global wakeword_listener
    if wakeword_listener is None:
        return
    try:
        wakeword_listener.stop()
    except Exception:
        pass
    wakeword_listener = None


def _restart_wakeword_listener(wakeword_cfg):
    _stop_wakeword_listener()
    return _start_wakeword_listener(wakeword_cfg)


galgame_window = None


def _open_galgame_window(character_name=""):
    """Open a transparent frameless PyQt5 window for Galgame display."""
    import sys as _sys
    import random
    import urllib.request
    import io as _io
    import json as _json
    import tempfile
    import os

    # 必须在主线程创建 QApplication
    from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QGraphicsDropShadowEffect
    from PyQt5.QtCore import Qt, QTimer, QPoint
    from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QFont, QPainterPath, QBrush, QPen

    # ── 加载角色数据 ──
    images_map = {}
    display_name = character_name
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/api/characters") as resp:
            data = _json.loads(resp.read().decode("utf-8"))
        chars = data.get("characters", [])
        char = None
        if character_name:
            char = next((c for c in chars if c["name"] == character_name), None)
        if not char and chars:
            char = chars[0]
        if char:
            images_map = char["images"]
            display_name = char["name"]
    except Exception as e:
        print(f"[galgame] 获取角色失败: {e}")
        return

    if not images_map:
        print("[galgame] 未找到角色或角色无图片")
        return

    # ── 创建应用 ──
    app = QApplication.instance() or QApplication(_sys.argv)

    # ── 预加载图片为 QPixmap（必须在 QApplication 之后）──
    pixmaps = {}  # {emotion: [QPixmap, ...]}

    def _load_pixmap(url_path):
        full_url = f"http://127.0.0.1:8000{url_path}"
        try:
            with urllib.request.urlopen(full_url) as resp:
                data = resp.read()
            img = QImage()
            img.loadFromData(data)
            if img.isNull():
                return None
            max_h = 650
            if img.height() > max_h:
                img = img.scaledToHeight(max_h, Qt.SmoothTransformation)
            return QPixmap.fromImage(img)
        except Exception:
            return None

    for emo, urls in images_map.items():
        pixmaps[emo] = []
        for url in urls:
            pm = _load_pixmap(url)
            if pm:
                pixmaps[emo].append(pm)

    class GalgameWindow(QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowFlags(
                Qt.FramelessWindowHint |
                Qt.WindowStaysOnTopHint |
                Qt.Tool
            )
            self.setAttribute(Qt.WA_TranslucentBackground)

            # 从 config 读取尺寸，默认 700x900
            _cfg_path = Path(__file__).resolve().parent / "config.json"
            _w, _h = 700, 900
            try:
                if _cfg_path.exists():
                    _cfg = json.loads(_cfg_path.read_text(encoding="utf-8"))
                    _l2d = _cfg.get("live2d", {})
                    _w = int(_l2d.get("galgame_width", 700))
                    _h = int(_l2d.get("galgame_height", 900))
            except Exception:
                pass
            self.win_w = max(400, min(_w, 2000))
            self.win_h = max(500, min(_h, 2000))
            self.setFixedSize(self.win_w, self.win_h)

            # 居中
            screen = app.primaryScreen().geometry()
            self.move(
                (screen.width() - self.win_w) // 2,
                (screen.height() - self.win_h) // 2
            )

            # 状态
            self.last_version = 0
            self.current_emotion = "neutral"
            self.reply_text = "..."
            self.portrait_pm = None
            self._drag_pos = None

            # 轮询定时器
            self.poll_timer = QTimer(self)
            self.poll_timer.timeout.connect(self._poll)
            self.poll_timer.start(500)

            # 初始立绘
            self._show_emotion("neutral")

            # 获取初始 version
            try:
                with urllib.request.urlopen("http://127.0.0.1:8000/api/live2d/state") as resp:
                    s = _json.loads(resp.read().decode("utf-8"))
                    self.last_version = s.get("audio_version", 0)
            except Exception:
                pass

        def _show_emotion(self, emotion):
            variants = pixmaps.get(emotion) or pixmaps.get("neutral")
            if not variants:
                return
            self.portrait_pm = random.choice(variants)
            self.current_emotion = emotion
            self.update()

        def _poll(self):
            try:
                with urllib.request.urlopen("http://127.0.0.1:8000/api/live2d/state") as resp:
                    state = _json.loads(resp.read().decode("utf-8"))
                ver = state.get("audio_version", 0)
                if ver > self.last_version:
                    self.last_version = ver
                    emo = state.get("emotion", "neutral")
                    if emo != self.current_emotion:
                        self._show_emotion(emo)
                    rt = state.get("reply_text", "")
                    if rt:
                        self.reply_text = rt
                        self.update()
                    self._play_audio(ver)
            except Exception:
                pass

        def _play_audio(self, version):
            def _do():
                try:
                    url = f"http://127.0.0.1:8000/api/live2d/audio?v={version}"
                    with urllib.request.urlopen(url) as resp:
                        data = resp.read()
                    tmp = os.path.join(tempfile.gettempdir(), "galgame_audio.wav")
                    with open(tmp, "wb") as f:
                        f.write(data)
                    import winsound
                    winsound.PlaySound(tmp, winsound.SND_FILENAME | winsound.SND_ASYNC)
                except Exception:
                    pass
            import threading
            threading.Thread(target=_do, daemon=True).start()

        # ── 绘制 ──
        def paintEvent(self, event):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setRenderHint(QPainter.SmoothPixmapTransform)

            # 立绘
            if self.portrait_pm and not self.portrait_pm.isNull():
                pm = self.portrait_pm
                # 底部对齐到对话框上方
                box_h = 130
                box_margin = 14
                avail_h = self.win_h - box_h - box_margin * 2
                # 缩放
                scaled = pm.scaled(
                    self.win_w - 40, avail_h,
                    Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                # 居中，底部留出对话框空间
                px = (self.win_w - scaled.width()) // 2
                py = avail_h - scaled.height() + box_margin
                painter.drawPixmap(px, py, scaled)

            # 对话框背景（圆角）
            box_x = 14
            box_y = self.win_h - 130 - 14
            box_w = self.win_w - box_x * 2
            box_h = 130
            radius = 16

            path = QPainterPath()
            path.addRoundedRect(float(box_x), float(box_y), float(box_w), float(box_h), radius, radius)

            # 半透明深色填充
            painter.fillPath(path, QBrush(QColor(15, 15, 30, 210)))

            # 细边框
            pen = QPen(QColor(255, 255, 255, 25))
            pen.setWidth(1)
            painter.setPen(pen)
            painter.drawPath(path)

            # 角色名
            painter.setPen(QColor(102, 204, 255))
            name_font = QFont("Microsoft YaHei", 14)
            name_font.setBold(True)
            painter.setFont(name_font)
            painter.drawText(box_x + 18, box_y + 28, display_name)

            # 对话文字
            painter.setPen(QColor(224, 224, 224))
            text_font = QFont("Microsoft YaHei", 12)
            painter.setFont(text_font)
            text_rect = painter.boundingRect(
                box_x + 18, box_y + 42, box_w - 36, box_h - 52,
                Qt.TextWordWrap | Qt.AlignLeft | Qt.AlignTop,
                self.reply_text
            )
            painter.drawText(text_rect, Qt.TextWordWrap | Qt.AlignLeft | Qt.AlignTop, self.reply_text)

            painter.end()

        # ── 拖拽 ──
        def mousePressEvent(self, event):
            if event.button() == Qt.LeftButton:
                self._drag_pos = event.globalPos() - self.frameGeometry().topLeft()
                event.accept()

        def mouseMoveEvent(self, event):
            if self._drag_pos and event.buttons() & Qt.LeftButton:
                self.move(event.globalPos() - self._drag_pos)
                event.accept()

        def mouseReleaseEvent(self, event):
            self._drag_pos = None

        # ── 右键关闭 ──
        def contextMenuEvent(self, event):
            self.close()

        # ── ESC 关闭 ──
        def keyPressEvent(self, event):
            if event.key() == Qt.Key_Escape:
                self.close()

    win = GalgameWindow()
    win.show()

    # 用 processEvents 循环代替 exec_()，避免阻塞线程
    while win.isVisible():
        app.processEvents()
        time.sleep(0.016)  # ~60fps


@app.post("/api/galgame/open")
async def open_galgame_window(request: Request):
    global galgame_window
    data = await request.json() if request.headers.get("content-type") == "application/json" else {}
    character = data.get("character", "")

    def _run():
        try:
            _open_galgame_window(character)
        except Exception as e:
            print(f"[galgame] 窗口错误: {e}")

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "ok"}


if __name__ == "__main__":
    if not frontend_dir.exists():
        frontend_dir.mkdir(parents=True, exist_ok=True)
        
    # Create an empty index.html if it doesn't exist
    index_file = frontend_dir / "index.html"
    if not index_file.exists():
        index_file.write_text("<h1>Loading...</h1>", encoding="utf-8")

    # 1. Start FastAPI server in a daemon thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # 2. Start wakeword listener if configured
    wakeword_cfg = _load_wakeword_config()
    _start_wakeword_listener(wakeword_cfg)

    # pywebview *must* run in the main thread
    window = webview.create_window(
        "Voice Assistant Control Panel", 
        "http://127.0.0.1:8000/", 
        width=1000, 
        height=800, 
        frameless=False,  # Can set to true for complete custom MD styling if wanted frame
        easy_drag=True
    )
    
    try:
        import pywebviewcli
    except ImportError:
        pass
        
    webview.start(debug=True)
    
    # When window closed, exit process
    sys.exit(0)

