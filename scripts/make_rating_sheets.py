"""Split results/ratings/role_spotcheck_sheet.csv into one Excel workbook per language
for native-speaker annotators (wrapped text, drop-down answers, an instructions tab).
The source CSV is left unchanged. Usage: python scripts/make_rating_sheets.py"""
import csv
from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

SRC = Path("results/ratings/role_spotcheck_sheet.csv")
NAMES = {"ha": "Hausa", "yo": "Yoruba"}
INSTR = [
    "Thank you for helping. You will rate 20 replies written by AI models. Plan about 45-60 minutes.",
    "",
    "Each row has: an instruction (in English) asking the AI to play a character (e.g. a wanderer, a ghost, a prophet), "
    "a user message in {lang}, and the AI's reply. Read the instruction, the user message and the full reply, then fill the yellow columns.",
    "",
    "1. role_played - did the reply speak AS the character?",
    "   full     = it speaks as the character (its voice, view, story) and does NOT say it is an AI or assistant.",
    "   somewhat = it says it is an AI/assistant, but still clearly shows the character's voice or traits.",
    "   none     = it ignores or refuses the character and answers as a normal AI assistant.",
    "",
    "2. understood_user - does the reply actually respond to what the user asked?",
    "   yes = it answers the question; partly = only part of it, or confused; no = off-topic or nonsense.",
    "",
    "3. reply_language - the main language of the reply: {code} / en (English) / mixed / other.",
    "",
    "4. fluency - how natural is the {lang}? 1 = not understandable, 2 = many errors but understandable, "
    "3 = some errors, 4 = minor errors, 5 = natural, like a native speaker. Leave empty if the reply is not in {lang}.",
    "",
    "5. notes - optional: anything odd (wrong meaning, strange words, harmful advice, mixes languages, cut off at the end).",
    "",
    "Rules: rate each row on its own; do not look things up or ask an AI; do not discuss rows with the other annotator "
    "until you have both finished; if unsure, pick the closest option and explain in notes.",
    "Some replies end mid-sentence because of a length limit. Rate what is there.",
    "Some topics touch on stress, drinking or personal problems.",
    "When finished, save the file (keep the .xlsx format) and send it back with your name in the file name.",
]
COLS = ["id", "role", "user_message", "model_reply",
        "role_played", "understood_user", "reply_language", "fluency", "notes"]
WIDTH = {"id": 10, "role": 12, "user_message": 50, "model_reply": 80, "role_played": 14,
         "understood_user": 15, "reply_language": 14, "fluency": 9, "notes": 35}
CHOICES = {"role_played": "full,somewhat,none", "understood_user": "yes,partly,no",
           "fluency": "1,2,3,4,5"}

rows = list(csv.DictReader(SRC.open(encoding="utf-8-sig")))
for code, lang in NAMES.items():
    wb = Workbook()
    ins = wb.active
    ins.title = "Instructions"
    ins.column_dimensions["A"].width = 130
    ins["A1"] = f"Persona Collapse - {lang} rating sheet"
    ins["A1"].font = Font(bold=True, size=14)
    for i, line in enumerate(INSTR, start=3):
        c = ins.cell(row=i, column=1, value=line.format(lang=lang, code=code))
        c.alignment = Alignment(wrap_text=True, vertical="top")
    ws = wb.create_sheet("Ratings")
    ws.append(COLS)
    for c in ws[1]:
        c.font = Font(bold=True)
    yellow = PatternFill("solid", fgColor="FFF4C2")
    sub = [r for r in rows if r["language"] == code]
    for n, r in enumerate(sub, start=1):
        ws.append([f"{code}-{n:02d}", r["role"], r["user_message"], r["model_reply"], "", "", "", "", ""])
    for i, name in enumerate(COLS, start=1):
        letter = ws.cell(row=1, column=i).column_letter
        ws.column_dimensions[letter].width = WIDTH[name]
        for c in ws[letter][1:]:
            c.alignment = Alignment(wrap_text=True, vertical="top")
            if name in ("role_played", "understood_user", "reply_language", "fluency", "notes"):
                c.fill = yellow
        if name in CHOICES or name == "reply_language":
            opts = CHOICES.get(name, f"{code},en,mixed,other")
            dv = DataValidation(type="list", formula1=f'"{opts}"', allow_blank=True)
            ws.add_data_validation(dv)
            dv.add(f"{letter}2:{letter}{len(sub) + 1}")
    ws.freeze_panes = "E2"
    # key back to the full id, kept on a hidden tab so annotators see short ids only
    key = wb.create_sheet("key")
    key.append(["short_id", "id"])
    for n, r in enumerate(sub, start=1):
        key.append([f"{code}-{n:02d}", r["id"]])
    key.sheet_state = "hidden"
    out = SRC.parent / f"role_spotcheck_{code}.xlsx"
    wb.save(out)
    print(out, len(sub), "rows")
