from pathlib import Path
import json
import threading
import time

from llm.config_manager import load_config
from llm.service import generate_chat
from speech import recorder
from speech.screenshot import attach_image_to_messages, capture_screenshot, load_screen_config
from stt import load_stt_config, transcribe_audio
from storage import HistoryStore
from tts.config import load_tts_config
from tts.gpt_sovits_client import synthesize_tts
from tts.player import apply_volume_wav, play_audio


SYSTEM_PROMPT = "你的名字是汐，是一个慵懒且有点三无的15岁少女，你和用户是朋友关系。你不需要对他使用敬语，可以对他开玩笑互损，说话要简短，要直白，像是正常朋友之间的交流，且不许在对话中使用括号进行动作神态等的描写例如“（打了个哈欠）我在呢”"


class VoiceChatWorkflow:
    def __init__(self, output_dir=Path("outputs"), system_prompt=SYSTEM_PROMPT):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.default_system_prompt = system_prompt
        self.system_prompt = self._load_system_prompt()
        self.history_store = HistoryStore(self.output_dir)
        self.processing_lock = threading.Lock()
        self.pending_screenshot = False
        self.screen_config = None

        self.llm_config = load_config()
        self.stt_config = load_stt_config(self.output_dir)
        self.tts_config = load_tts_config(self.output_dir)
        self.recording_config = self._load_recording_config()

        self.last_reply_text_path = self.output_dir / "latest_reply.txt"
        self.last_reply_audio_path = self.output_dir / "latest_reply.wav"
        self.last_reply_scaled_audio_path = self.output_dir / "latest_reply_scaled.wav"

    def _load_system_prompt(self):
        config_path = Path(__file__).resolve().parents[1] / "config.json"
        if config_path.exists():
            try:
                data = json.loads(config_path.read_text(encoding="utf-8"))
                prompt = data.get("system_prompt", "").strip()
                if prompt:
                    return prompt
            except Exception:
                pass
        return self.default_system_prompt

    def reload_config(self):
        self.llm_config = load_config()
        self.stt_config = load_stt_config(self.output_dir)
        self.tts_config = load_tts_config(self.output_dir)
        self.recording_config = self._load_recording_config()
        self.system_prompt = self._load_system_prompt()

    def _load_recording_config(self):
        config_path = Path(__file__).resolve().parents[1] / "config.json"
        default = {
            "device_index": None,
            "vad": {
                "enabled": True,
                "silence_seconds": 1.2,
                "min_record_seconds": 0.8,
                "aggressiveness": 2,
                "frame_ms": 30,
            }
        }
        if not config_path.exists():
            return default
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return default
            cfg = data.get("recording", {})
            if not isinstance(cfg, dict):
                return default
            device_index = cfg.get("device_index")
            if not isinstance(device_index, int):
                device_index = None
            vad_cfg = cfg.get("vad", {})
            if not isinstance(vad_cfg, dict):
                vad_cfg = {}
            merged = {
                "device_index": device_index,
                "vad": {**default["vad"], **vad_cfg},
            }
            return merged
        except Exception:
            return default

    def _start_vad_monitor(self, vad_config):
        if not vad_config.get("enabled"):
            return
        silence_seconds = float(vad_config.get("silence_seconds", 1.2))
        min_record_seconds = float(vad_config.get("min_record_seconds", 0.8))

        def _monitor():
            start_time = recorder.recording_state.get("vad_start_time", time.time())
            while recorder.recording_state.get("is_recording"):
                time.sleep(0.1)
                if time.time() - start_time < min_record_seconds:
                    continue
                last_voice = recorder.recording_state.get("vad_last_voice_time", start_time)
                if time.time() - last_voice >= silence_seconds:
                    self.stop_and_process()
                    break

        thread = threading.Thread(target=_monitor, daemon=True)
        thread.start()

    def start_or_stop_recording(self, with_screenshot=False):
        if self.processing_lock.locked():
            print("[system] 正在处理上一段，请稍后再录音。")
            return

        if recorder.recording_state.get("is_recording"):
            if self.pending_screenshot != with_screenshot:
                if self.pending_screenshot:
                    print("[system] 当前是截图录音，请再按 Alt+2 结束并处理。")
                else:
                    print("[system] 当前是普通录音，请再按 Alt+1 结束并处理。")
                return
            self.stop_and_process()
            return

        self.pending_screenshot = with_screenshot
        vad_config = self.recording_config.get("vad", {})
        input_device_index = self.recording_config.get("device_index")
        if with_screenshot:
            recorder.start_recording(
                stop_hint="再次按 Alt+2 停止并处理",
                vad_config=vad_config,
                input_device_index=input_device_index,
            )
        else:
            recorder.start_recording(
                stop_hint="再次按 Alt+1 停止并处理",
                vad_config=vad_config,
                input_device_index=input_device_index,
            )
        self._start_vad_monitor(vad_config)

    def stop_and_process(self):
        if not self.processing_lock.acquire(blocking=False):
            print("[system] Busy. Please wait...")
            return

        with_screenshot = self.pending_screenshot
        self.pending_screenshot = False

        def _runner():
            try:
                self._process_pipeline(with_screenshot=with_screenshot)
            finally:
                self.processing_lock.release()

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()

    def force_stop_only(self):
        if recorder.recording_state.get("is_recording"):
            recorder.stop_recording()
            self.pending_screenshot = False
            print("[system] 已停止录音(未发送)。")
            return
        print("[system] 当前没有进行中的录音。")

    def clear_history(self):
        self.history_store.clear(delete_screenshots=True)
        print("[system] New session started.")

    def _build_messages(self, history):
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(history)
        return messages

    def _load_screen_config_once(self):
        if self.screen_config is None:
            self.screen_config = load_screen_config(self.output_dir)
        return self.screen_config

    def _process_pipeline(self, with_screenshot=False):
        if not recorder.stop_recording():
            return

        text = transcribe_audio(stt_config=self.stt_config)
        if not text or not text.strip():
            print("[system] 转录为空，已忽略。")
            return

        screenshot_path = None
        if with_screenshot:
            screen_config = self._load_screen_config_once()
            screenshot_path = capture_screenshot(screen_config, self.output_dir)

        meta = None
        if screenshot_path:
            meta = {"screenshot": str(Path(screenshot_path).as_posix())}

        self.history_store.append("user", text, meta=meta)
        history = self.history_store.load()
        messages = self._build_messages(history)

        if screenshot_path:
            provider = self.llm_config.provider.lower()
            if provider in {"openai", "openai_compat", "api", "deepseek"}:
                messages = attach_image_to_messages(messages, screenshot_path)
            else:
                print("[warning] 当前提供商不支持图片，已忽略截图。")

        reply = generate_chat(messages, self.llm_config)
        if not reply:
            return

        self.history_store.append("assistant", reply)
        self.last_reply_text_path.write_text(reply, encoding="utf-8")

        audio_path = synthesize_tts(reply, self.tts_config, self.last_reply_audio_path)
        if audio_path:
            adjusted_path = apply_volume_wav(
                audio_path,
                self.tts_config.get("volume", 1.0),
                self.last_reply_scaled_audio_path,
            )
            play_audio(adjusted_path)
