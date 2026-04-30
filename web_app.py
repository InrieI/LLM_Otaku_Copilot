import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import webview
import threading
import json
from pathlib import Path
from pydantic import BaseModel
import sys
import os
import time

from workflows import VoiceChatWorkflow
from speech import recorder
import keyboard

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
hotkey_handles = []
outputs_dir = Path(__file__).resolve().parent / "outputs"
wakeword_dataset_dir = outputs_dir / "wakeword_dataset"
wakeword_models_dir = outputs_dir / "wakeword_models"
wakeword_training_state = {"status": "idle", "message": "", "name": ""}
wakeword_sample_state = {"path": None, "label": None, "name": None}
live2d_models_dir = Path(__file__).resolve().parent / "live2d-models"
live2d_state = {"emotion": "neutral", "reply_text": "", "audio_version": 0}
workflow.live2d_state = live2d_state

# Serve static files for frontend
app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

# Serve Live2D model files
if live2d_models_dir.exists():
    app.mount("/live2d-models", StaticFiles(directory=str(live2d_models_dir)), name="live2d-models")

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
    wakeword_started = _restart_wakeword_listener(wakeword_cfg)
    run_assistant(enable_hotkeys=not wakeword_started)
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

def run_server():
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="error")

def _enable_hotkeys():
    global hotkey_handles
    if hotkey_handles:
        return
    # We redefine hotkeys here to ensure they still work while the UI is open.
    hotkey_handles = [
        keyboard.add_hotkey("alt+1", lambda: workflow.start_or_stop_recording(with_screenshot=False)),
        keyboard.add_hotkey("alt+2", lambda: workflow.start_or_stop_recording(with_screenshot=True)),
        keyboard.add_hotkey("alt+3", workflow.force_stop_only),
        keyboard.add_hotkey("alt+0", workflow.clear_history),
    ]
    print("[system] Assistant ready. Hotkeys active in background.")


def _disable_hotkeys():
    global hotkey_handles
    if not hotkey_handles:
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
    with_screenshot = bool(wakeword_cfg.get("with_screenshot", False))

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

    # 2. Start the core Voice Assistant workflow hotkeys or wakeword
    wakeword_cfg = _load_wakeword_config()
    wakeword_started = _start_wakeword_listener(wakeword_cfg)
    run_assistant(enable_hotkeys=not wakeword_started)

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

