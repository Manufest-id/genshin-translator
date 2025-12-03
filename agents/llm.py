import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.getenv("AI_PROVIDER", "google").lower()  # "openai" | "google"
API_KEY = os.getenv("AI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.4"))

if not API_KEY:
    raise RuntimeError("AI_API_KEY is not set in .env")


def _openai_call(system: str, user: str, temperature: float = TEMPERATURE) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": OPENAI_MODEL,
        "temperature": float(temperature),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    if not r.ok:
        raise RuntimeError(f"OpenAI error {r.status_code}: {r.text}")
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()


def _gemini_call(system: str, user: str, temperature: float = TEMPERATURE) -> str:
    import requests, json, os

    model = GEMINI_MODEL
    headers = {"x-goog-api-key": API_KEY, "Content-Type": "application/json"}

    # v1 (camelCase) payload — works for 1.5 family
    payload_v1 = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {"temperature": float(temperature)},
    }
    url_v1 = (
        f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent"
    )
    r = requests.post(url_v1, headers=headers, json=payload_v1, timeout=120)
    if r.ok:
        data = r.json()
        parts = []
        for cand in data.get("candidates", []):
            for p in cand.get("content", {}).get("parts", []):
                t = p.get("text", "")
                if t:
                    parts.append(t)
        out = "".join(parts).strip()
        if out:
            return out

    # If v1 failed (e.g., 400 unknown systemInstruction), try v1beta (snake_case) — needed for 2.x family
    payload_beta = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generation_config": {"temperature": float(temperature)},
    }
    url_beta = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    rb = requests.post(url_beta, headers=headers, json=payload_beta, timeout=120)
    if not rb.ok:
        raise RuntimeError(f"Gemini error {rb.status_code}: {rb.text}")

    data = rb.json()
    parts = []
    for cand in data.get("candidates", []):
        for p in cand.get("content", {}).get("parts", []):
            t = p.get("text", "")
            if t:
                parts.append(t)
    out = "".join(parts).strip()
    if not out:
        raise RuntimeError(f"Gemini returned empty response: {json.dumps(data)[:400]}")
    return out


def llm_call(system: str, user: str, temperature: float = TEMPERATURE) -> str:
    if PROVIDER == "openai":
        return _openai_call(system, user, temperature)
    if PROVIDER == "google":
        return _gemini_call(system, user, temperature)
    raise RuntimeError(f"Unsupported AI_PROVIDER: {PROVIDER}")


# Optional tiny smoke test utility
def translate_simple(text_cn: str, tgt_lang: str = "en") -> str:
    system = "You are a professional translator. Output only the translation."
    user = f"Translate from Chinese (Simplified) to {'English' if tgt_lang=='en' else 'Indonesian'}:\n{text_cn}"
    return llm_call(system, user)
