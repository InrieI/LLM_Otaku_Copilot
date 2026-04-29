from pathlib import Path
import json


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


ROOT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.json"


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


def _normalize_path(value):
    if not isinstance(value, str):
        return ""
    cleaned = value.strip().strip('"').strip("'")
    return cleaned.replace("\\", "/")


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


def _format_aux_list(aux_list):
    if not aux_list:
        return "(无)"
    return "\n".join(f"  {idx + 1}) {path}" for idx, path in enumerate(aux_list))


def load_tts_config(output_dir=Path("outputs")):
    _ = output_dir
    root_config = _load_root_config()

    raw = root_config.get("tts")
    has_existing_file = isinstance(raw, dict)
    if not has_existing_file:
        raw = DEFAULT_TTS_CONFIG

    config = _normalize_tts_config(raw)

    config["ref_audio_path"] = _normalize_path(config.get("ref_audio_path", ""))
    config["aux_ref_audio_paths"] = [
        _normalize_path(path) for path in config.get("aux_ref_audio_paths", [])
    ]
    config["volume"] = _clamp_volume(config.get("volume", 1.0))

    if not has_existing_file:
        root_config["tts"] = config
        _save_root_config(root_config)
        
    return config

