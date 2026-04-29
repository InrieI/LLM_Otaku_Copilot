from llm.providers import (
    generate_deepseek,
    generate_deepseek_messages,
    generate_gemini,
    generate_ollama,
    generate_openai_compat,
    generate_openai_compat_messages,
)


def generate(prompt, config):
    provider = config.provider.lower()

    if provider == "ollama":
        return generate_ollama(prompt, config)
    if provider in {"openai", "openai_compat", "api"}:
        return generate_openai_compat(prompt, config)
    if provider == "deepseek":
        return generate_deepseek(prompt, config)
    if provider == "gemini":
        return generate_gemini(prompt, config)

    raise ValueError(f"不支持的 AI 提供商: {config.provider}")


def generate_chat(messages, config, temperature=0.7):
    provider = config.provider.lower()

    if provider in {"openai", "openai_compat", "api"}:
        return generate_openai_compat_messages(messages, config, temperature)
    if provider == "deepseek":
        return generate_deepseek_messages(messages, config, temperature)

    prompt = messages_to_prompt(messages)
    if provider == "ollama":
        return generate_ollama(prompt, config)
    if provider == "gemini":
        return generate_gemini(prompt, config)

    raise ValueError(f"不支持的 AI 提供商: {config.provider}")


def messages_to_prompt(messages):
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
