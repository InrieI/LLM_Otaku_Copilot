import requests


def generate(prompt, config):
    url = f"{config.ollama_url}/api/generate"
    payload = {
        "model": config.model,
        "prompt": prompt,
        "stream": False,
    }
    response = requests.post(url, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    return data.get("response", "").strip()
