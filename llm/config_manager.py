from dataclasses import dataclass
from pathlib import Path
import getpass
import json

import requests


@dataclass
class LLMConfig:
    provider: str
    model: str
    ollama_url: str
    openai_base_url: str
    deepseek_base_url: str
    api_key: str


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.json"
DEFAULT_CONFIG = {
    "provider": "",
    "model": "",
    "ollama_url": "http://localhost:11434",
    "openai_base_url": "https://api.openai.com/v1",
    "deepseek_base_url": "https://api.deepseek.com",
    "providers": {
        "gemini": {
            "keys": [],
            "last_used": 0,
        },
        "openai_compat": {
            "keys": [],
            "last_used": 0,
        },
        "deepseek": {
            "keys": [],
            "last_used": 0,
        },
    },
}


def _load_json_config():
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(json.dumps(DEFAULT_CONFIG, indent=2), encoding="utf-8")
        print(f"[system] 已生成默认配置文件: {CONFIG_PATH.name}")

    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("配置文件不是有效的 JSON 对象")
        return data
    except Exception as exc:
        print(f"[warning] 读取配置失败，改用默认值: {exc}")
        return {}


def _save_json_config(data):
    CONFIG_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _normalize_config(file_config):
    config = json.loads(json.dumps(DEFAULT_CONFIG))

    for key in ("provider", "model", "ollama_url", "openai_base_url", "deepseek_base_url"):
        if file_config.get(key):
            config[key] = file_config[key]

    file_providers = file_config.get("providers", {})
    for provider_name in config["providers"]:
        provider_cfg = file_providers.get(provider_name, {})
        keys = provider_cfg.get("keys", [])
        if not isinstance(keys, list):
            keys = []
        last_used = provider_cfg.get("last_used", 0)
        if not isinstance(last_used, int):
            last_used = 0
        config["providers"][provider_name]["keys"] = keys
        config["providers"][provider_name]["last_used"] = last_used

    legacy_key = file_config.get("api_key")
    if legacy_key and isinstance(legacy_key, str):
        provider = config.get("provider")
        if provider in config["providers"]:
            if legacy_key not in config["providers"][provider]["keys"]:
                config["providers"][provider]["keys"].append(legacy_key)

    # Preserve non-LLM top-level nodes (for example: stt/tts) when rewriting config.json.
    for key, value in file_config.items():
        if key not in config:
            config[key] = value

    return config


def _needs_setup(config):
    provider = config.get("provider")
    if provider not in ("ollama", "gemini", "openai_compat", "deepseek"):
        return True
    if provider == "ollama":
        return not config.get("model")
    if provider in config.get("providers", {}):
        return not config["providers"][provider]["keys"]
    return True


def _mask_key(key):
    if len(key) <= 8:
        return f"{key[:2]}****"
    return f"{key[:4]}****{key[-4:]}"


def save_config(new_config_dict):
    """
    接收来自前端的完整配置字典并保存，
    同时保留非LLM节点的配置。
    """
    current = _load_json_config()
    for k, v in new_config_dict.items():
        current[k] = v
    _save_json_config(current)

def list_ollama_models(ollama_url):
    url = f"{ollama_url}/api/tags"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()
    models = [model.get("name", "") for model in data.get("models", [])]
    return [name for name in models if name]

def load_config(interactive=False):
    # 彻底弃用 interactive 交互式终端输入
    file_config = _load_json_config()
    config = _normalize_config(file_config)

    provider = config.get("provider", "")
    model = config.get("model", "")
    ollama_url = config.get("ollama_url", DEFAULT_CONFIG["ollama_url"])
    openai_base_url = config.get("openai_base_url", DEFAULT_CONFIG["openai_base_url"])
    deepseek_base_url = config.get("deepseek_base_url", DEFAULT_CONFIG["deepseek_base_url"])

    api_key = ""
    if provider in config.get("providers", {}):
        keys = config["providers"][provider].get("keys", [])
        last_used = config["providers"][provider].get("last_used", 0)
        if keys:
            api_key = keys[last_used] if 0 <= last_used < len(keys) else keys[0]

    return LLMConfig(
        provider=provider,
        model=model,
        ollama_url=ollama_url,
        openai_base_url=openai_base_url,
        deepseek_base_url=deepseek_base_url,
        api_key=api_key,
    )

