# streamlit_app.py
import os
import io
import json
import tempfile
import pandas as pd
import streamlit as st

from agents.llm import llm_call
from agents.persona_learner import learn_personas_from_csv, save_personas, load_personas

# Optional: if you have this module. If not, comment the import and the call below.
from agents.normalizer import normalize_indonesian

st.set_page_config(page_title="Game Translator", page_icon="🗣️", layout="wide")

# Defaults that match your spreadsheets
DEFAULT_CN = "简体中文 zh-CN"
DEFAULT_ID = "印尼语 id-ID"
DEFAULT_CHAR_COL = "character_id"  # we’ll synthesize this in temp CSV during training

PERSONA_DIR = "data"
os.makedirs(PERSONA_DIR, exist_ok=True)


# ---------- helpers ----------
def read_any_dataframe(uploaded_file) -> pd.DataFrame:
    """Read CSV or Excel into a DataFrame."""
    name = uploaded_file.name.lower()
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    # sep=None + engine='python' lets pandas sniff delimiter (, ; \t)
    return pd.read_csv(uploaded_file, sep=None, engine="python", encoding="utf-8")


def df_to_xlsx_bytes(df: pd.DataFrame) -> bytes:
    """Return an in-memory XLSX binary for download."""
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    return bio.getvalue()


def build_system():
    return (
        "You are a professional game localization translator.\n"
        "Preserve placeholders/tags exactly (e.g., {NICKNAME}, {PLAYER_NAME}, <color=...> ... </color>).\n"
        "Follow the character persona and mimic tone and quirks.\n"
        "Output only the translation."
    )


def persona_to_lines(persona: dict, tgt_lang: str) -> str:
    """Flatten the PersonaProfile fields into short guidance lines for the LLM."""
    lines = []
    if persona.get("tone"):
        lines.append(f"Tone: {persona['tone']}")
    if persona.get("formality"):
        lines.append(f"Formality: {persona['formality']}")
    if persona.get("pronouns"):
        lines.append(f"Preferred pronouns/register: {persona['pronouns']}")
    if persona.get("punctuation_habits"):
        lines.append(f"Punctuation habits: {persona['punctuation_habits']}")
    if persona.get("quirks"):
        lines.append(f"Quirks: {', '.join([q for q in persona['quirks'] if q])}")
    if persona.get("lexical_preferences"):
        lines.append(
            f"Lexical preferences: {', '.join([w for w in persona['lexical_preferences'] if w])}"
        )
    rules_key = "style_rules_id" if tgt_lang == "id" else "style_rules_en"
    if persona.get(rules_key):
        lines.append(f"Style rules: {', '.join([r for r in persona[rules_key] if r])}")
    return "\n".join(lines) if lines else "Use neutral tone."


def build_user(src_cn: str, persona: dict, tgt_lang: str):
    persona_lines = persona_to_lines(persona, tgt_lang)
    return f"""
Translate from Chinese (Simplified) to {'Indonesian' if tgt_lang=='id' else 'English'}.

Persona guidance:
{persona_lines}

Keep placeholders/tags verbatim: tokens like {{NICKNAME}}, {{PLAYER_NAME}}, and tags such as <color=...> ... </color>.
Do not translate or remove them. Keep lines concise and suitable for UI/dialog.

Text:
{src_cn}

Output only the translation.
""".strip()


# ---------- UI ----------
st.title("🎮 Game Translator — Persona-aware (Pydantic schema)")

tabs = st.tabs(["📚 Train Persona", "📝 Translate"])

# -------------------- TRAIN TAB --------------------
with tabs[0]:
    st.subheader("Upload Training Data (Chinese + Indonesian/English)")
    train_file = st.file_uploader(
        "Training file (.xlsx / .csv)", type=["xlsx", "xls", "csv"], key="train"
    )

    # Choose target language for the persona to learn
    tcol1, tcol2, tcol3 = st.columns(3)
    with tcol1:
        target_lang = st.selectbox("Target language", options=["id", "en"], index=0)
    with tcol2:
        train_cn_col = st.text_input("Chinese column name", DEFAULT_CN)
    with tcol3:
        train_tgt_col = st.text_input(
            f"{'Indonesian' if target_lang=='id' else 'English'} column name",
            DEFAULT_ID if target_lang == "id" else "English",
        )

    train_char_id = st.text_input(
        "Character ID (constant for this training set)", "new_char_x"
    )
    max_lines = st.slider("Max lines to sample for learning", 50, 3000, 300, 50)

    persona_json_path = os.path.join(
        PERSONA_DIR, f"personas_learned_{target_lang}.json"
    )

    if st.button("Learn Persona", type="primary", disabled=train_file is None):
        if train_file is None:
            st.error("Please upload a training file.")
        else:
            try:
                df = read_any_dataframe(train_file)
                if train_cn_col not in df.columns or train_tgt_col not in df.columns:
                    st.error(f"Missing columns. Found: {list(df.columns)}")
                else:
                    progress = st.progress(0.0, text="Preparing data…")

                    # Synthesize a minimal training CSV with a fixed character_id column
                    tmp_df = pd.DataFrame(
                        {
                            DEFAULT_CHAR_COL: train_char_id,
                            "zh_cn": df[train_cn_col].astype(str),
                            "tgt": df[train_tgt_col].astype(str),
                        }
                    )
                    progress.progress(0.2, text="Sampling lines…")

                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".csv", dir="."
                    ) as tf:
                        tmp_df.to_csv(tf.name, index=False, encoding="utf-8")
                        progress.progress(0.4, text="Inferring persona (LLM)…")

                        profiles = learn_personas_from_csv(
                            csv_path=tf.name,
                            char_col=DEFAULT_CHAR_COL,
                            src_col="zh_cn",
                            tgt_col="tgt",
                            target_lang=target_lang,  # 'id' or 'en'
                            max_lines_per_char=max_lines,
                        )

                    os.unlink(tf.name)
                    progress.progress(0.8, text="Saving learned persona…")

                    # Merge with existing file (so you can keep multiple characters)
                    existing = {}
                    if os.path.exists(persona_json_path):
                        try:
                            existing = load_personas(persona_json_path)
                        except Exception:
                            existing = {}
                    existing.update(profiles)
                    save_personas(existing, persona_json_path)

                    st.session_state.setdefault("personas", {})
                    st.session_state["personas"][target_lang] = existing

                    progress.progress(1.0, text="Done")
                    st.success(f"Persona learned and saved to {persona_json_path}")
                    st.code(json.dumps(profiles, ensure_ascii=False, indent=2))
            except Exception as e:
                st.exception(e)

# -------------------- TRANSLATE TAB --------------------
with tabs[1]:
    st.subheader("Translate Testing Data")
    test_file = st.file_uploader(
        "Testing file (.xlsx / .csv)", type=["xlsx", "xls", "csv"], key="test"
    )

    # Choose target language for translation (this must match the learned persona language)
    lcol1, lcol2, lcol3 = st.columns(3)
    with lcol1:
        trans_lang = st.selectbox(
            "Target language", options=["id", "en"], index=0, key="trans_lang"
        )
    with lcol2:
        test_cn_col = st.text_input("Chinese column name", DEFAULT_CN, key="test_cn")
    with lcol3:
        test_tgt_col = st.text_input(
            f"{'Indonesian' if trans_lang=='id' else 'English'} target column name",
            DEFAULT_ID if trans_lang == "id" else "English",
            key="test_tgt",
        )

    # Persona usage
    personas_all = st.session_state.get("personas", {})
    if os.path.exists(os.path.join(PERSONA_DIR, f"personas_learned_{trans_lang}.json")):
        try:
            personas_all[trans_lang] = load_personas(
                os.path.join(PERSONA_DIR, f"personas_learned_{trans_lang}.json")
            )
        except Exception:
            pass

    available_ids = sorted(list((personas_all.get(trans_lang) or {}).keys()))
    st.caption(
        f"Learned personas available ({trans_lang}): "
        f"{', '.join(available_ids) if available_ids else '(none)'}"
    )

    no_training = st.checkbox("No training data for this character (neutral persona)")
    use_learned_persona = st.checkbox(
        "Use learned persona (if available)", value=not no_training
    )
    test_char_id = st.text_input(
        "Character ID to use at translation", "new_char_x", key="test_charid"
    )

    # Ancillary opts
    use_kbbi = st.checkbox("Apply KBBI normalization (udah→sudah, etc.)", value=True)
    overwrite = st.checkbox("Overwrite existing translations", value=False)
    sleep_s = st.number_input(
        "Sleep between calls (sec)", min_value=0.0, max_value=2.0, value=0.0, step=0.1
    )

    # Optional custom mapping for KBBI normalization
    kbbi_map_file = st.file_uploader(
        "Optional: Custom KBBI mapping (JSON)", type=["json"]
    )
    custom_map = None
    if kbbi_map_file is not None:
        try:
            custom_map = json.loads(kbbi_map_file.read().decode("utf-8"))
            st.caption(f"Loaded {len(custom_map)} mappings.")
        except Exception as e:
            st.warning(f"Invalid KBBI JSON: {e}")

    # Persona preview area
    persona = {}
    if not no_training and use_learned_persona:
        persona = (personas_all.get(trans_lang) or {}).get(test_char_id, {}) or {}
        if persona:
            with st.expander(
                f"Persona preview: {test_char_id} ({trans_lang})", expanded=True
            ):
                st.json(persona, expanded=False)
            st.success(f"Using learned persona: **{test_char_id}**")
        else:
            st.warning(
                "No learned persona found for this character/language. Using neutral persona."
            )
            persona = {}

    run_btn = st.button("Run Translation", type="primary", disabled=test_file is None)

    if run_btn:
        try:
            df = read_any_dataframe(test_file)
            if test_cn_col not in df.columns:
                st.error(f"Missing column: {test_cn_col}")
                st.stop()
            if test_tgt_col not in df.columns:
                df[test_tgt_col] = ""

            progress = st.progress(0.0, text="Starting…")
            total = len(df)
            system_prompt = build_system()
            filled = 0

            for idx, row in df.iterrows():
                src = str(row[test_cn_col]) if pd.notna(row[test_cn_col]) else ""
                if not src.strip():
                    pct = (idx + 1) / total
                    progress.progress(pct, text=f"{idx+1}/{total} (skipped empty)")
                    continue

                existing = str(row[test_tgt_col]) if pd.notna(row[test_tgt_col]) else ""
                if existing.strip() and not overwrite:
                    pct = (idx + 1) / total
                    progress.progress(pct, text=f"{idx+1}/{total} (kept existing)")
                    continue

                # Choose persona: neutral if no_training or none available
                persona_used = persona if (not no_training and persona) else {}

                user_prompt = build_user(src, persona_used, tgt_lang=trans_lang)
                try:
                    out = llm_call(system_prompt, user_prompt)
                    if use_kbbi and trans_lang == "id":
                        try:
                            out = normalize_indonesian(out, custom_map=custom_map)
                        except Exception:
                            # Fallback if normalizer not present
                            pass
                    df.at[idx, test_tgt_col] = out
                    filled += 1
                except Exception as e:
                    df.at[idx, test_tgt_col] = existing  # keep whatever was there
                    st.write(f"Row {idx+1}: ERROR {e}")

                if sleep_s > 0:
                    import time as _t

                    _t.sleep(sleep_s)

                pct = (idx + 1) / total
                progress.progress(pct, text=f"{idx+1}/{total} (filled {filled})")

            st.success(f"Done. Filled {filled} cell(s).")
            xbytes = df_to_xlsx_bytes(df)
            st.download_button(
                "Download result (.xlsx)",
                data=xbytes,
                file_name="translated_output.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception as e:
            st.exception(e)
