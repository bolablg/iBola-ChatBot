# app/guardrails.py
import re, unicodedata
from typing import Tuple

def _normalize(text: str) -> str:
    # lower, strip accents, collapse spaces
    t = text.lower().strip()
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")
    t = re.sub(r"\s+", " ", t)
    return t

# Minimal multilingual coverage for your scope
KEYWORDS = {
    # scope anchors
    "bolaji","resume","cv","parcours","profil","career","carriere","work","travail","job","poste","project","projet",
    "experience","experiences","skill","skills","competence","competences","gozem",
    # domains
    "data","donnees","analytics","analytique","statistics","statistiques","ai","ia","machine learning",
    "python","sql","google","bigquery","clickhouse","tableau","looker","dataform","chatbot",
    # education
    "master","maitrise","maitrise","licence","bachelor","phd","doctorat","diplome","diploma",
    "etudes","etude","formation","education","universite","ecole","school","university"
}

# Targeted patterns (degrees, studies, roles)
PATTERNS = [
    r"\b(master|maitrise|maitrise|licence|bachelor|phd|doctorat|dipl[o0]me|degree)\b",
    r"\b(etudes|formation|education|universite|ecole|school|university)\b",
    r"\b(role|poste|position|titre|title)\b",
    r"\b(skill[s]?|competence[s]?|experience[s]?)\b",
    r"\b(projet[s]?|project[s]?)\b",
]

def is_in_scope(message: str, chat_history: str = "") -> bool:
    """
    Returns True if the message is about Bolaji's professional life.
    Uses normalized multilingual keywords + patterns + soft heuristics.
    """
    text = _normalize(message)
    hist = _normalize(chat_history)

    # 1) Keyword hits
    kw_hits = sum(1 for k in KEYWORDS if f" {k} " in f" {text} ")

    # 2) Regex pattern hits
    pat_hits = sum(1 for p in PATTERNS if re.search(p, text))

    # 3) Conversation continuity: if history is already in-scope, be lenient
    hist_hits = sum(1 for k in KEYWORDS if f" {k} " in f" {hist} ")
    continuity = hist_hits > 0

    score = kw_hits + pat_hits + (1 if continuity else 0)

    # Threshold: 1+ is enough; continuity lets short follow-ups through
    return score >= 1

# Example:
# is_in_scope("as tu un master ?")  -> True
# is_in_scope("parle-moi de tes etudes") -> True
# is_in_scope("quel est ton film prefere ?") -> False