from __future__ import annotations
from typing import List, Dict, Any
import json
import math
import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from .llm import llm_call


# ---------- Pydantic schema to keep the LLM honest ----------
class PersonaProfile(BaseModel):
    character_id: str
    target_lang: str = Field(pattern="^(en|id)$")
    tone: str
    quirks: List[str] = []
    pronouns: str | None = (
        None  # e.g., "aku/kamu" or "saya/anda" for ID; "I/you" or formal patterns for EN
    )
    formality: str | None = None  # informal / neutral / formal
    punctuation_habits: str | None = None  # exclamation use, ellipses, etc.
    lexical_preferences: List[str] = []  # recurring word choices / catchphrases
    style_rules_en: List[str] = []  # rules when target is English
    style_rules_id: List[str] = []  # rules when target is Indonesian
    notes: str | None = None


# ---------- core function ----------
def learn_personas_from_csv(
    csv_path: str,
    char_col: str,
    src_col: str,
    tgt_col: str,
    target_lang: str,  # "en" or "id"
    max_lines_per_char: int = 30,
    max_chars: int | None = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Read a CSV with at least: character_id, source_text, translated_text.
    For each character: sample up to N target-language lines, and ask the LLM to infer a persona profile.
    Returns a dict: {character_id: profile_dict}
    """
    df = pd.read_csv(csv_path)
    need = {char_col, src_col, tgt_col}
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    # Optional: limit total characters processed (useful for quick tests)
    if max_chars:
        df = df.head(max_chars)

    # Group lines by character
    groups = df.groupby(char_col)
    results: Dict[str, Dict[str, Any]] = {}

    for character_id, g in groups:
        # Collect up to N translated lines for this character
        tgt_lines = g[tgt_col].dropna().astype(str).tolist()[:max_lines_per_char]
        if not tgt_lines:
            continue

        sample_block = "\n".join(f"- {line}" for line in tgt_lines)

        # Build a compact, schema-driven prompt
        system = (
            "You are a localization style analyst. "
            "Given several translated dialog lines from ONE game character, infer their speaking style."
        )
        user = f"""
You will be given up to {max_lines_per_char} lines **already translated** to { 'English' if target_lang=='en' else 'Indonesian' }.
From these, infer a concise persona profile capturing tone, quirks, formality, pronoun choices, punctuation habits, and recurring lexical preferences.

Return STRICT JSON that validates this Pydantic model:

{PersonaProfile.schema_json(indent=2)}

Character ID: {character_id}
Target language code: {target_lang}

Translated sample lines:
{sample_block}

Important:
- Base your analysis ONLY on these translated lines.
- Keep it short and concrete; no generic fluff.
- If a field is unknown, leave it empty or minimal.
- Output ONLY the JSON object, nothing else.
""".strip()

        raw = llm_call(system, user, temperature=0.2)

        # Be defensive: find the first { .. } block and parse
        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            obj = json.loads(raw[start:end])
            obj["character_id"] = str(character_id)
            obj["target_lang"] = target_lang
            profile = PersonaProfile(**obj)  # validate
            results[character_id] = profile.model_dump()
        except (json.JSONDecodeError, ValidationError, Exception) as e:
            # If LLM returns junk, store a minimal profile so the pipeline still works
            results[character_id] = PersonaProfile(
                character_id=str(character_id),
                target_lang=target_lang,
                tone="neutral",
                quirks=[],
            ).model_dump()

    return results


# ---------- helper to save/load ----------
def save_personas(profiles: Dict[str, Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)


def load_personas(path: str) -> Dict[str, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
