"""
Learn a single character persona from two columns:
- Chinese source:  简体中文 zh-CN
- Indonesian translation:  印尼语 id-ID

It treats the entire file as ONE character (constant character_id).
Outputs: data/personas_learned_id.json
"""

import os
import json
import pandas as pd
from agents.persona_learner import learn_personas_from_csv, save_personas

# ==== CONFIG (edit these for your file) ====
INPUT_PATH = "train_data.xlsx"  # can be .xlsx/.xls or .csv
CHAR_ID = "new_char_x"  # constant id for this unreleased character
CN_COL = "简体中文 zh-CN"  # source column name
ID_COL = "印尼语 id-ID"  # translated column name (used for style learning)
TARGET_LANG = "id"  # "id" or "en" (matches the translation column you learn from)
MAX_LINES = 200  # sample size per character for learning
OUTPUT_JSON = f"data/personas_learned_{TARGET_LANG}.json"
TMP_DIR = "io"
TMP_CSV = os.path.join(TMP_DIR, "_tmp_single_char.csv")
# ==========================================

os.makedirs("data", exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)


def load_sheet(path: str) -> pd.DataFrame:
    """Robust loader for Excel or CSV. Sniffs CSV delimiter if needed."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    # For CSV, let pandas sniff comma/semicolon
    return pd.read_csv(path, sep=None, engine="python", encoding="utf-8")


def main():
    # 1) Load the file
    df = load_sheet(INPUT_PATH)

    # 2) Minimal validation
    missing = [c for c in (CN_COL, ID_COL) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # 3) Build a tiny CSV with only what the learner needs + constant character_id
    tmp = pd.DataFrame(
        {
            "character_id": CHAR_ID,
            "zh_cn": df[CN_COL].astype(str),
            "id_id": df[ID_COL].astype(str),
        }
    )
    tmp.to_csv(TMP_CSV, index=False, encoding="utf-8")

    # 4) Learn persona from the target column (Indonesian or English)
    profiles = learn_personas_from_csv(
        csv_path=TMP_CSV,
        char_col="character_id",
        src_col="zh_cn",
        tgt_col="id_id",
        target_lang=TARGET_LANG,
        max_lines_per_char=MAX_LINES,
    )

    # 5) Save learned persona(s)
    save_personas(profiles, OUTPUT_JSON)
    print(f"Saved persona to {OUTPUT_JSON}")
    print(json.dumps(profiles, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
