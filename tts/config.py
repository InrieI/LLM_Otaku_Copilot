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


def _load_json(path, default_value):
    if not path.exists():
        return default_value
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default_value


def _save_json(path, data):
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


def _prompt_if_missing(config, key, label):
    value = str(config.get(key, "")).strip()
    if value:
        return config
    print(f"[setup] {label} 为空，请输入。")
    config[key] = input("> ").strip()
    return config


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


def load_tts_config(output_dir=Path("outputs")):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tts_config_path = output_dir / "tts_config.json"

    has_existing_file = tts_config_path.exists()
    raw = _load_json(tts_config_path, DEFAULT_TTS_CONFIG)
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

    _save_json(tts_config_path, config)
    return config
