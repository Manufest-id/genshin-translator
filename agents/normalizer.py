# agents/normalizer.py
import re
from typing import Dict, Optional

# A tiny starter map; extend as you like (or pass a custom map).
DEFAULT_KBBI_MAP: Dict[str, str] = {
    "udah": "sudah",
}

_WORD_RE = re.compile(r"\b([\w\-']+)\b", re.UNICODE)


def _match_casing(src: str, dst: str) -> str:
    if src.isupper():
        return dst.upper()
    if src.istitle():
        return dst.capitalize()
    return dst


def normalize_indonesian(text: str, custom_map: Optional[Dict[str, str]] = None) -> str:
    """
    Replace colloquial words with KBBI-ish forms, preserving basic casing.
    Only exact token matches (word-boundary) are replaced.
    """
    mapping = {**DEFAULT_KBBI_MAP}
    if custom_map:
        mapping.update({k.lower(): v for k, v in custom_map.items()})

    def repl(m: re.Match) -> str:
        w = m.group(1)
        low = w.lower()
        if low in mapping:
            return _match_casing(w, mapping[low])
        return w

    return _WORD_RE.sub(repl, text)
