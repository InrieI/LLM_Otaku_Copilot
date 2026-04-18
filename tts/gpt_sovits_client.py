from pathlib import Path
import base64

import requests


def _decode_base64(text):
    try:
        return base64.b64decode(text, validate=True)
    except Exception:
        return None


def _save_audio_from_response(response, output_path):
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


def synthesize_tts(text, tts_config, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base_url = tts_config.get("base_url", "http://127.0.0.1:9880").rstrip("/")
    url = f"{base_url}/tts"

    payload = dict(tts_config)
    payload["text"] = text
    payload.pop("base_url", None)

    response = requests.post(url, json=payload, timeout=120)
    if response.status_code != 200:
        print(f"[error] TTS failed: {response.status_code} {response.text}")
        return None

    saved = _save_audio_from_response(response, output_path)
    if not saved:
        print("[error] Failed to parse TTS response.")
    return saved
