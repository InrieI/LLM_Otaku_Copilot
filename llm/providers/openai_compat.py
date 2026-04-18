import requests


def _headers(api_key):
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def generate(prompt, config, temperature=0.7):
    if not config.api_key:
        raise ValueError("OPENAI_API_KEY 没有设置")

    url = f"{config.openai_base_url}/chat/completions"
    payload = {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }

    response = requests.post(url, json=payload, headers=_headers(config.api_key), timeout=60)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def generate_messages(messages, config, temperature=0.7):
    if not config.api_key:
        raise ValueError("OPENAI_API_KEY 没有设置")

    url = f"{config.openai_base_url}/chat/completions"
    payload = {
        "model": config.model,
        "messages": messages,
        "temperature": temperature,
    }

    response = requests.post(url, json=payload, headers=_headers(config.api_key), timeout=60)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()
