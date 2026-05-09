from pathlib import Path
import re
import pandas as pd


DATA_ROOT = Path("C:\Users\krsna\OneDrive\Desktop\aarabi research\data\dementia_speech_research\data\raw\Pitt")


def clean_chat_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)          # remove <...>
    text = re.sub(r"\[.*?\]", " ", text)          # remove [/] [//] [: ...]
    text = re.sub(r"&[-\w]+", " ", text)          # remove &-uh, &-um
    text = re.sub(r"\+<", " ", text)              # remove overlap marker
    text = re.sub(r"\[\+.*?\]", " ", text)        # remove [+ gram], [+ exc]
    text = re.sub(r"[/()]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def parse_metadata(lines):
    age = None
    sex = None
    group = None
    mmse = None

    for line in lines:
        if line.startswith("@ID:") and "|PAR|" in line:
            parts = line.strip().split("|")

            # Example:
            # @ID: eng|Pitt|PAR|58;|female|Control||Participant|30||
            if len(parts) > 8:
                age = parts[3].replace(";", "").strip()
                sex = parts[4].strip()
                group = parts[5].strip()
                mmse = parts[8].strip()

    return age, sex, group, mmse


def extract_participant_text(lines):
    par_lines = []

    for line in lines:
        if line.startswith("*PAR:"):
            text = line.replace("*PAR:", "").strip()
            par_lines.append(clean_chat_text(text))

    return " ".join(par_lines)


def build_dataset():
    rows = []

    for label in ["Control", "Dementia"]:
        folder = DATA_ROOT / label / "cookie"

        for cha_file in folder.glob("*.cha"):
            lines = cha_file.read_text(encoding="utf-8", errors="ignore").splitlines()

            age, sex, group, mmse = parse_metadata(lines)
            transcript = extract_participant_text(lines)

            rows.append({
                "file_id": cha_file.stem,
                "label": label,
                "age": age,
                "sex": sex,
                "mmse": mmse,
                "transcript": transcript,
                "cha_path": str(cha_file),
                "audio_path": str(cha_file.with_suffix(".mp3"))
            })

    df = pd.DataFrame(rows)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df.to_csv("data/processed/transcripts_dataset.csv", index=False)

    print("Dataset created successfully!")
    print(df.head())
    print(df["label"].value_counts())


if __name__ == "__main__":
    build_dataset()