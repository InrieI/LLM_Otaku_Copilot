from dataclasses import dataclass
from pathlib import Path
import json
import os
import requests
import getpass


# @dataclass 是一个装饰器，自动为这个类生成 __init__ 方法
# 简单说：定义了一个"配置数据结构"，用来存放 LLM 的配置信息
@dataclass
class LLMConfig:
    provider: str              # 使用哪个 AI 提供商："ollama"、"openai_compat"、"gemini"
    model: str                 # 模型名字，比如 "qwen2:7b"、"gpt-4"、"gemini-2.0-flash"
    ollama_url: str            # Ollama 服务的地址，比如 "http://localhost:11434"
    openai_base_url: str       # OpenAI 兼容 API 的地址
    api_key: str               # API 密钥（OpenAI、Gemini 需要，Ollama 不需要）


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


def _select_api_key(provider_name, provider_cfg):
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

        api_key, last_used, keys = _select_api_key(provider, config["providers"][provider])
        config["providers"][provider]["keys"] = keys
        config["providers"][provider]["last_used"] = last_used

    _save_json_config(config)
    return config


def load_config(interactive=False):
    """
    从 config.json 读取配置
    如果 config.json 没有配置，就用默认值
    需要时可进入终端交互式配置
    
    支持的提供商：
    - ollama: 本地 Ollama 服务
    - openai_compat: OpenAI 兼容的 API（包括 OpenAI、Claude 等）
    - gemini: Google Gemini API
    
    例子（config.json）：
    - "provider": "ollama"
    - "model": "qwen2:7b"
    - "provider": "gemini", "model": "gemini-2.0-flash"
    """
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


def list_ollama_models(ollama_url): 
    """
    连接到 Ollama 服务器，获取所有可用的本地模型列表
    
    参数: ollama_url - Ollama 的地址，比如 "http://localhost:11434"
    返回: 模型名字的列表，比如 ["qwen2:7b", "llama2"]
    """
    url = f"{ollama_url}/api/tags"  # Ollama 的 API 端点
    response = requests.get(url, timeout=10)  # 发送 HTTP 请求获取数据
    response.raise_for_status()  # 如果请求失败，抛出异常
    data = response.json()  # 将返回的 JSON 转成 Python 字典
    
    # 从返回的数据中提取模型名字
    models = [model.get("name", "") for model in data.get("models", [])]
    # 过滤掉空字符串，只保留有效的模型名
    return [name for name in models if name]


def generate(prompt, config: LLMConfig):
    """
    根据配置选择使用哪个 AI 提供商，然后生成回答
    
    参数:
    - prompt: 用户的问题或输入
    - config: LLMConfig 配置对象
    
    返回: AI 的回答（字符串）
    """
    provider = config.provider.lower()  # 将提供商名字转小写
    
    # 根据提供商类型选择不同的处理函数
    if provider == "ollama":
        return _generate_ollama(prompt, config)
    if provider in {"openai", "openai_compat", "api"}:
        return _generate_openai_compat(prompt, config)
    if provider == "gemini":
        return _generate_gemini(prompt, config)
    
    # 如果提供商名字没有被识别，抛出错误
    raise ValueError(f"不支持的 AI 提供商: {config.provider}")


def generate_chat(messages, config: LLMConfig, temperature=0.7):
    """
    根据配置生成聊天回复（支持带 role 的历史消息）

    messages: [{"role": "system|user|assistant", "content": "..."}, ...]
    """
    provider = config.provider.lower()

    if provider in {"openai", "openai_compat", "api"}:
        return _generate_openai_compat_messages(messages, config, temperature)

    prompt = _messages_to_prompt(messages)
    if provider == "ollama":
        return _generate_ollama(prompt, config)
    if provider == "gemini":
        return _generate_gemini(prompt, config)

    raise ValueError(f"不支持的 AI 提供商: {config.provider}")


def _messages_to_prompt(messages):
    lines = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            prefix = "System"
        elif role == "assistant":
            prefix = "Assistant"
        else:
            prefix = "User"
        lines.append(f"{prefix}: {content}")
    lines.append("Assistant:")
    return "\n".join(lines)


def _generate_ollama(prompt, config: LLMConfig):
    """
    调用本地 Ollama 模型生成回答
    
    原理：通过 HTTP 请求将问题发送给 Ollama 服务，获取回答
    """
    # Ollama 生成文本的 API 地址
    url = f"{config.ollama_url}/api/generate"
    
    # 构造请求数据（Python 字典）
    payload = {
        "model": config.model,      # 指定使用哪个模型
        "prompt": prompt,           # 用户的问题
        "stream": False             # False = 等待完整回答后返回；True = 流式返回（字一个字返回）
    }
    
    # 发送 POST 请求到 Ollama
    response = requests.post(url, json=payload, timeout=60)
    response.raise_for_status()  # 如果请求失败，抛出异常
    result = response.json()  # 解析返回的 JSON
    
    # 从返回的数据中提取回答，去掉前后空格
    return result.get("response", "").strip()


def _generate_openai_compat(prompt, config: LLMConfig):
    """
    调用 OpenAI 兼容的 API（比如 OpenAI、Gemini 等）生成回答
    
    这些 API 需要密钥（Authorization token）来验证身份
    """
    # 检查是否设置了 API 密钥
    if not config.api_key:
        raise ValueError("OPENAI_API_KEY 没有设置")

    # OpenAI 的聊天完成 API 地址
    url = f"{config.openai_base_url}/chat/completions"
    
    # 构造请求头（HTTP 请求的信息）
    headers = {
        "Authorization": f"Bearer {config.api_key}",  # 身份验证：发送 API 密钥
        "Content-Type": "application/json"            # 告诉服务器发送的是 JSON 数据
    }
    
    # 构造请求数据
    payload = {
        "model": config.model,      # 指定使用哪个模型（比如 "gpt-4"）
        "messages": [               # 消息列表（OpenAI API 需要这个格式）
            {"role": "user", "content": prompt}  # 用户说的话
        ],
        "temperature": 0.7,         # 生成的随机性：0.7 = 标准值（0 = 确定性强，1.0 = 随机性强）
    }
    
    # 发送 POST 请求到 OpenAI API
    response = requests.post(url, json=payload, headers=headers, timeout=60)
    response.raise_for_status()  # 如果请求失败，抛出异常
    data = response.json()  # 解析返回的 JSON
    
    # 从返回的数据中提取回答
    # data["choices"][0]["message"]["content"] = 第一个回答选项的内容
    return data["choices"][0]["message"]["content"].strip()


def _generate_openai_compat_messages(messages, config: LLMConfig, temperature=0.7):
    if not config.api_key:
        raise ValueError("OPENAI_API_KEY 没有设置")

    url = f"{config.openai_base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.model,
        "messages": messages,
        "temperature": temperature,
    }

    response = requests.post(url, json=payload, headers=headers, timeout=60)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def _generate_gemini(prompt, config: LLMConfig):
    """
    调用 Google Gemini API 生成回答
    
    Gemini 使用不同的 API 端点和请求格式
    """
    # 检查是否设置了 API 密钥
    if not config.api_key:
        raise ValueError("GEMINI_API_KEY 没有设置")

    # Gemini 的生成文本 API 地址
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{config.model}:generateContent"
    
    # 构造请求数据（Gemini 的格式）
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}  # 用户的问题
                ]
            }
        ]
    }
    
    # 添加 API 密钥到 URL 参数（Gemini 用这种方式传递密钥）
    params = {"key": config.api_key}
    
    # 发送 POST 请求到 Gemini API
    response = requests.post(url, json=payload, params=params, timeout=60)
    response.raise_for_status()  # 如果请求失败，抛出异常
    data = response.json()  # 解析返回的 JSON
    
    # 从返回的数据中提取回答
    # Gemini 的格式：candidates[0].content.parts[0].text
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


# 这是"主程序"，当你直接运行这个文件时会执行
if __name__ == "__main__":
    # 加载配置
    cfg = load_config(interactive=True)
    
    print(f"[system] 使用提供商: {cfg.provider}")
    print(f"[system] 使用模型: {cfg.model}")
    
    # 测试：问 AI 一个简单的问题
    print()
    answer = generate("你好！请简单自我介绍一下。", cfg)
    print(f"[assistant] {answer}")
