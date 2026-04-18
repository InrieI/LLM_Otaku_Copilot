from llm.config_manager import LLMConfig, list_ollama_models, load_config
from llm.service import generate, generate_chat


__all__ = ["LLMConfig", "load_config", "list_ollama_models", "generate", "generate_chat"]


if __name__ == "__main__":
    cfg = load_config(interactive=True)
    print(f"[system] 使用提供商: {cfg.provider}")
    print(f"[system] 使用模型: {cfg.model}")
    print()
    answer = generate("你好！请简单自我介绍一下。", cfg)
    print(f"[assistant] {answer}")
