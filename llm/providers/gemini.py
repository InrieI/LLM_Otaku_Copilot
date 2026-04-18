import requests


def generate(prompt, config):
    if not config.api_key:
        raise ValueError("GEMINI_API_KEY 没有设置")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{config.model}:generateContent"
    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}],
            }
        ]
    }
    params = {"key": config.api_key}

    response = requests.post(url, json=payload, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()
