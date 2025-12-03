from agents.persona_learner import learn_personas_from_csv, save_personas

CSV_PATH = "mini_dialog.csv"
CHAR_COL = "character_id"
SRC_COL = "zh_cn"
TGT_COL = "id_id"  # we’re learning style from Indonesian lines

profiles = learn_personas_from_csv(
    csv_path=CSV_PATH,
    char_col=CHAR_COL,
    src_col=SRC_COL,
    tgt_col=TGT_COL,
    target_lang="id",
    max_lines_per_char=10,
)

print("LEARNED PROFILES:\n", profiles)
save_personas(profiles, "personas_learned_id.json")
print("Saved to personas_learned_id.json")
