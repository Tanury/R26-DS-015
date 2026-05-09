import re


def clean_chat_text(text: str) -> str:
    text = re.sub(r"(\S+)\s*\[:\s*([^\]]+)\]", r"\2", text)
    text = text.replace("<", " ").replace(">", " ")
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"&[-\w]+", " ", text)
    text = text.replace("+<", " ")
    text = re.sub(r"\(([^)]*)\)", r"\1", text)
    text = re.sub(r"[^a-zA-Z0-9'\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def extract_par_text_from_cha(content: str) -> str:
    lines = content.splitlines()
    par_lines = []

    for line in lines:
        line = line.strip()
        if line.startswith("*PAR:"):
            utterance = line.replace("*PAR:", "", 1).strip()
            cleaned = clean_chat_text(utterance)
            if cleaned:
                par_lines.append(cleaned)

    return " ".join(par_lines)