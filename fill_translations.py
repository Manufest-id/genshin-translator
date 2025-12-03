import os
import json
import time
import argparse
import pandas as pd
from agents.llm import llm_call
from agents.normalizer import normalize_indonesian


# ---------------- I/O helpers ----------------
def load_sheet(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path, sep=None, engine="python", encoding="utf-8")


def save_sheet(df: pd.DataFrame, path: str) -> None:
    ext = os.path.splitext(path)[1].lower()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if ext in (".xlsx", ".xls"):
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False, encoding="utf-8")


def default_output_path(input_path: str) -> str:
    base, ext = os.path.splitext(input_path)
    return f"{base}_out{ext or '.xlsx'}"


# ---------------- prompting helpers ----------------
def build_system() -> str:
    return (
        "You are a professional game localization translator.\n"
        "Preserve placeholders/tags exactly (e.g., {NICKNAME}, {PLAYER_NAME}, <color=...> ... </color>).\n"
        "Follow the character persona and mimic tone and quirks.\n"
        "Output only the translation."
    )


def build_user(src_cn: str, persona: dict, tgt_lang: str) -> str:
    tone = persona.get("tone", "neutral")
    quirks = ", ".join(persona.get("quirks", [])) or "none"
    rules = (
        ", ".join(
            persona.get("style_rules_id" if tgt_lang == "id" else "style_rules_en", [])
        )
        or "none"
    )

    return f"""
Translate from Chinese (Simplified) to {'Indonesian' if tgt_lang=='id' else 'English'}.

Persona:
- Tone: {tone}
- Quirks: {quirks}
- Rules: {rules}

Untranslatable tokens (preserve verbatim): patterns like {{NICKNAME}}, {{PLAYER_NAME}},
and tags like <color=...> ... </color>. Do not remove or translate them.

Text:
{src_cn}

Requirements:
1) Preserve placeholders/tags exactly.
2) Keep sentences concise for UI.
3) Match persona tone/quirks.
4) Output only translated text.
""".strip()


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(
        description="Auto-fill the Indonesian translation column next to the Chinese source, using a learned persona."
    )
    ap.add_argument("--input", required=True, help="Path to .xlsx/.xls/.csv")
    ap.add_argument("--output", help="Output path (default: <input>_out.<ext>)")
    ap.add_argument(
        "--char-id", required=True, help="Character id used when learning the persona"
    )
    ap.add_argument(
        "--cn-col", default="简体中文 zh-CN", help="Chinese source column name"
    )
    ap.add_argument(
        "--id-col", default="印尼语 id-ID", help="Indonesian target column name"
    )
    ap.add_argument(
        "--persona-json",
        default="data/personas_learned_id.json",
        help="Path to learned persona JSON",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite non-empty target cells (default: fill only blanks)",
    )
    ap.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep between API calls (rate-limit safety)",
    )
    ap.add_argument(
        "--autosave-every",
        type=int,
        default=0,
        help="Autosave progress every N filled rows to <output>.work",
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        default=20,
        help="Print progress every N processed rows",
    )
    # NEW:
    ap.add_argument(
        "--kbbi",
        action="store_true",
        help="Apply KBBI-style normalization on Indonesian output",
    )
    ap.add_argument("--kbbi-json", help="Path to custom JSON mapping for normalization")
    args = ap.parse_args()

    # Resolve output paths
    output_path = args.output or default_output_path(args.input)
    work_path = output_path + ".work"

    # Load persona
    if os.path.exists(args.persona_json):
        with open(args.persona_json, "r", encoding="utf-8") as f:
            personas = json.load(f)
        persona = personas.get(args.char_id, {})
    else:
        print(
            f"WARNING: persona JSON not found: {args.persona_json}. Using neutral persona."
        )
        persona = {}

    # Load custom KBBI map (optional)
    custom_map = None
    if args.kbbi_json and os.path.exists(args.kbbi_json):
        with open(args.kbbi_json, "r", encoding="utf-8") as f:
            custom_map = json.load(f)

    # Load spreadsheet
    df = load_sheet(args.input)

    # Ensure columns
    if args.cn_col not in df.columns:
        raise ValueError(f"Missing column: {args.cn_col}")
    if args.id_col not in df.columns:
        df[args.id_col] = ""

    total = len(df)
    filled = 0
    processed = 0
    system_prompt = build_system()

    try:
        for i, row in df.iterrows():
            processed += 1

            src = str(row[args.cn_col]) if pd.notna(row[args.cn_col]) else ""
            if not src.strip():
                continue

            tgt_existing = str(row[args.id_col]) if pd.notna(row[args.id_col]) else ""
            if tgt_existing.strip() and not args.overwrite:
                continue

            user_prompt = build_user(src, persona, tgt_lang="id")

            try:
                out = llm_call(system_prompt, user_prompt)
            except Exception as e:
                print(f"[{processed}/{total}] ERROR: {e}")
                continue

            if args.kbbi:
                out = normalize_indonesian(out, custom_map=custom_map)

            df.at[i, args.id_col] = out
            filled += 1

            # autosave checkpoint
            if args.autosave_every and filled % args.autosave_every == 0:
                save_sheet(df, work_path)
                print(f"[autosave] Wrote checkpoint: {work_path} (filled {filled})")

            if args.sleep > 0:
                time.sleep(args.sleep)

            if args.progress_every and processed % args.progress_every == 0:
                print(f"Progress: filled {filled} rows")

        # final save
        save_sheet(df, output_path)
        if os.path.exists(work_path):
            try:
                os.remove(work_path)
            except Exception:
                pass
        print(f"Done. Filled {filled} cell(s). Wrote: {output_path}")

    except KeyboardInterrupt:
        print("\nInterrupted. Saving checkpoint...")
        save_sheet(df, work_path)
        print(f"[checkpoint] Progress saved to: {work_path} (filled {filled})")
        print(
            "Re-run the script later; it will skip already-filled cells unless you pass --overwrite."
        )


if __name__ == "__main__":
    main()
