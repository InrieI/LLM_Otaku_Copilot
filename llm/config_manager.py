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
    api_key: str


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.json"
DEFAULT_CONFIG = {
    "provider": "",
    "model": "",
    "ollama_url": "http://localhost:11434",
    "openai_base_url": "https://api.openai.com/v1",
    "providers": {
        "gemini": {
            "keys": [],
            "last_used": 0,
        },
        "openai_compat": {
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

    for key in ("provider", "model", "ollama_url", "openai_base_url"):
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

    return config


def _needs_setup(config):
    provider = config.get("provider")
    if provider not in ("ollama", "gemini", "openai_compat"):
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


def _prompt_text(title, default_value):
    value = input(f"{title} (回车使用默认: {default_value})\n> ").strip()
    return value if value else default_value


def _prompt_api_key():
    try:
        key = getpass.getpass("请输入新的 API Key (输入时不会显示，可直接粘贴):\n> ").strip()
        if key:
            return key
        print("[warning] 未输入内容，改用可见输入")
    except (EOFError, KeyboardInterrupt):
        print("[warning] 隐藏输入不可用，改用可见输入")

    return input("请输入新的 API Key (会显示在屏幕上):\n> ").strip()


def _prompt_provider(config):
    providers = [
        ("ollama", "本地 Ollama"),
        ("gemini", "Gemini API"),
        ("openai_compat", "OpenAI / GitHub Models (兼容)"),
    ]
    current = config.get("provider")
    default_index = next((i for i, item in enumerate(providers) if item[0] == current), None)

    print("请选择提供商：")
    for idx, item in enumerate(providers, start=1):
        suffix = " (默认)" if default_index is not None and idx - 1 == default_index else ""
        print(f"{idx}) {item[1]}{suffix}")

    choice = input("> ").strip()
    if choice == "" and default_index is not None:
        return providers[default_index][0]
    if choice.isdigit():
        index = int(choice) - 1
        if 0 <= index < len(providers):
            return providers[index][0]

    print("[warning] 输入无效，已使用默认提供商")
    return providers[default_index][0] if default_index is not None else "ollama"


def list_ollama_models(ollama_url):
    url = f"{ollama_url}/api/tags"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()
    models = [model.get("name", "") for model in data.get("models", [])]
    return [name for name in models if name]


def _select_ollama_model(config):
    try:
        models = list_ollama_models(config["ollama_url"])
    except Exception as exc:
        print(f"[warning] 无法检测本地模型: {exc}")
        return config

    if not models:
        print("[warning] 没检测到本地模型，请先用 Ollama 拉取模型")
        return config

    current = config.get("model")
    default_index = models.index(current) if current in models else 0

    print("请选择本地模型：")
    for idx, name in enumerate(models, start=1):
        suffix = " (默认)" if idx - 1 == default_index else ""
        print(f"{idx}) {name}{suffix}")

    choice = input("> ").strip()
    if choice == "":
        config["model"] = models[default_index]
        return config
    if choice.isdigit():
        index = int(choice) - 1
        if 0 <= index < len(models):
            config["model"] = models[index]
            return config

    print("[warning] 输入无效，已使用默认模型")
    config["model"] = models[default_index]
    return config


def _prompt_delete_key(keys):
    if not keys:
        print("[warning] 没有可删除的 Key")
        return None

    print("选择要删除的 Key：")
    for idx, key in enumerate(keys, start=1):
        print(f"{idx}) {_mask_key(key)}")
    choice = input("> ").strip()
    if choice.isdigit():
        index = int(choice) - 1
        if 0 <= index < len(keys):
            return index
    print("[warning] 输入无效")
    return None


def _select_api_key(provider_cfg):
    keys = provider_cfg.get("keys", [])
    last_used = provider_cfg.get("last_used", 0)

    while True:
        if keys:
            print("请选择 API Key：")
            for idx, key in enumerate(keys, start=1):
                suffix = " (默认)" if idx - 1 == last_used else ""
                print(f"{idx}) {_mask_key(key)}{suffix}")
            print("A) 添加新 Key")
            print("D) 删除 Key")

            choice = input("> ").strip().lower()
            if choice == "" and 0 <= last_used < len(keys):
                return keys[last_used], last_used, keys
            if choice.isdigit():
                index = int(choice) - 1
                if 0 <= index < len(keys):
                    return keys[index], index, keys
            if choice == "a":
                new_key = _prompt_api_key()
                if new_key:
                    keys.append(new_key)
                    return new_key, len(keys) - 1, keys
                print("[warning] API Key 不能为空")
                continue
            if choice == "d":
                delete_index = _prompt_delete_key(keys)
                if delete_index is None:
                    continue
                del keys[delete_index]
                if not keys:
                    print("[system] 已删除，Key 列表为空")
                    last_used = 0
                    continue
                if last_used >= len(keys):
                    last_used = 0
                print("[system] 已删除")
                continue

            print("[warning] 请输入有效选项")
        else:
            new_key = _prompt_api_key()
            if new_key:
                keys.append(new_key)
                return new_key, 0, keys
            print("[warning] API Key 不能为空")


def _interactive_setup(config):
    if not _needs_setup(config):
        print("[system] 已有配置，是否修改？(y/N)")
        answer = input("> ").strip().lower()
        if answer != "y":
            return config

    provider = _prompt_provider(config)
    config["provider"] = provider

    if provider == "ollama":
        config = _select_ollama_model(config)
    else:
        default_model = "gemini-2.0-flash" if provider == "gemini" else "gpt-4o-mini"
        config["model"] = _prompt_text("请输入模型名称", config.get("model") or default_model)

        if provider == "openai_compat":
            default_base_url = config.get("openai_base_url") or DEFAULT_CONFIG["openai_base_url"]
            config["openai_base_url"] = _prompt_text("请输入 API Base URL", default_base_url)

        api_key, last_used, keys = _select_api_key(config["providers"][provider])
        config["providers"][provider]["keys"] = keys
        config["providers"][provider]["last_used"] = last_used

    _save_json_config(config)
    return config


def load_config(interactive=False):
    file_config = _load_json_config()
    config = _normalize_config(file_config)

    if interactive:
        config = _interactive_setup(config)

    provider = config.get("provider", "")
    model = config.get("model", "")
    ollama_url = config.get("ollama_url", DEFAULT_CONFIG["ollama_url"])
    openai_base_url = config.get("openai_base_url", DEFAULT_CONFIG["openai_base_url"])

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
        api_key=api_key,
    )
