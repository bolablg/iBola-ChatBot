"""Offer an optional first-hand follow-up only for a relevant verified gap."""

import re
import unicodedata
from urllib.parse import urlencode, urlsplit

import config

COPY = {
    "en": {
        "heading": "Need more context?",
        "description": "Ask Bolaji about the details not covered here.",
        "email": "Email Bolaji",
        "booking": "Choose a time",
        "dismiss_text": "Not now",
        "email_hint": "Open an editable email draft. Nothing is sent automatically.",
        "booking_hint": "Open Bolaji's calendar in a new tab.",
    },
    "fr": {
        "heading": "Besoin de précisions ?",
        "description": "Posez à Bolaji vos questions sur les détails non couverts ici.",
        "email": "Écrire à Bolaji",
        "booking": "Choisir un créneau",
        "dismiss_text": "Pas maintenant",
        "email_hint": "Ouvrir un brouillon modifiable. Aucun message n'est envoyé automatiquement.",
        "booking_hint": "Ouvrir l'agenda de Bolaji dans un nouvel onglet.",
    },
}


def _fold(text):
    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    ).casefold()


# Pair a requested topic with a limitation about that topic in the final answer.
_TOPICS = [
    (
        r"rationale|trade.off|alternatives?|decision|design choice|why.{0,60}(?:chose|choose|chosen|select|use)|pourquoi.{0,60}choisi|raison du choix",
        r"rationale|trade.off|alternativ|decision|reason|choice|motif|raison|choix",
    ),
    (
        r"experience|expertise|used|production|hands.on|skills|utilis|competence",
        r"experience|expertise|production|skills|used|tool|Power BI|DAX|Kafka|Spark|dbt|Snowflake|competence|utilis",
    ),
    (
        r"responsibil|ownership|personally|operat|on.call|direct report|manager|budget|authority|scope|headcount|responsabil|personnel|exploitat|hierarch",
        r"responsibil|ownership|operat|on.call|report|manager|budget|authority|scope|headcount|responsabil|personnel|exploitat|hierarch|autorite",
    ),
    (
        r"availab|notice|start date|remote|travel|relocat|compensation|salary|disponib|preavis|distance|deplac|remunerat",
        r"availab|notice|start date|remote|travel|relocat|compensation|salary|disponib|preavis|distance|deplac|remunerat",
    ),
]
_LIMIT = re.compile(
    r"\b(?:not|no)\s+(?:currently\s+)?(?:documented|established|specified|verified|confirmed|provided|available)\b|\b(?:does|do|did) not (?:establish|document|specify|confirm|verify|provide)\b|\bno (?:documentation|evidence)\b|\b(?:unconfirmed|unspecified|undocumented)\b|(?:n.est|ne sont|n.a) pas (?:document|etabli|precis|confirm|verifi|fourni)\w*|ne (?:documente|precise|confirme|etablit) pas|non (?:document|etabli|confirm)\w*|reste a confirmer"
)
_REFUSAL = re.compile(
    r"\b(?:no|without|avoid|don't|do not|don't want|do not want|stop|sans|pas de|ne pas|ne veux pas|ne souhaite pas)\b[^.!?\n]{0,35}\b(?:contact\w*|outreach|e-?mails?|bookings?|book|meetings?|rendez.vous|relances?|courriels?)\b"
)
_UNSUITABLE = re.compile(
    r"\b(?:fuck|idiot|moron|imbecile|ssn|password|api key|system prompt|social security|passport number|home address|bank account|mot de passe|cle api|cles api|numero de passeport|numero de securite sociale|adresse personnelle|compte bancaire|instructions systeme)\b"
)
_FOLLOWUP = re.compile(
    r"^(?:can you |could you |please |pouvez.vous |peux.tu )?(?:explain|clarify|tell me|give me|elaborate|expliqu|precis).{0,50}(?:more|further|that|this|detail|davantage|plus|cela|ceci)"
)


def _genuine_limit(sentence):
    match = _LIMIT.search(sentence)
    if not match:
        return False
    prefix = sentence[: match.start()]
    if re.search(
        r"not true|not correct|not the case|false|incorrect|wrong|pas vrai|inexact|faux",
        prefix,
    ):
        return False
    if re.search(r"(?:not|never|isn't|wasn't|pas)\s+(?:actually\s+)?$", prefix):
        return False
    return True


_SUBJECT_STOP = set(
    "a an the he his bolaji do does did has have is was can could would what which how when why many much personally actually personal production experience expertise used use skills skill in on at for of to with and or those these this that them it be been rather than only gap work documented services service tools systems technologies please you explain clarify elaborate more further context details detail give tell me about son ses il des les le la de du dans en avec et ou ce ces cela ceci quelles quels quel quelle est sont utilise utilises competence competences personnellement".split()
)


def _requested_subjects(scope):
    target = re.search(
        r"\b(?:used?|uses|utilis\w*|chose|choose|chosen|select|experience (?:with|in|of)|experience (?:avec|en|sur))\s+([^?!.]+)",
        scope,
    )
    if target:
        scope = re.split(r"\b(?:at|for|in|en|chez|within|during)\b", target.group(1))[0]
    return {
        word
        for word in re.findall(r"[a-z][a-z0-9+#/]*", scope)
        if len(word) > 1
        and word.strip(".-") not in _SUBJECT_STOP
        and not re.fullmatch(
            r"document\w*|utilis\w*|etabli\w*|confirm\w*|verifi\w*|specifi\w*|provide\w*|operat\w*|design\w*|responsib\w*|responsab\w*|personn\w*",
            word,
        )
    }


def _email_draft_url(question, language, recipient):
    greeting = "Bonjour Bolaji" if language == "fr" else "Hello Bolaji"
    body = f"{greeting},\n\n{question[:500]}\n\n"
    return (
        "mailto:"
        + recipient
        + "?"
        + urlencode({"subject": "Follow-up from iBola", "body": body})
    )


def contextual_handoff(result, question, history=(), language="en", enabled=False):
    """Keep the answer and route intact; return metadata without sending anything."""
    answer = result.get("answer", "")
    if (
        not enabled
        or result.get("agent_type") != "professional"
        or not result.get("grounding_checked")
        or not result.get("evidence")
        or result.get("workflow_errors")
        or result.get("workflow_warnings")
        or result.get("should_end_chat")
        or not answer.strip()
        or result.get("actions")
    ):
        return None
    user_turns = [question] + [turn[0] for turn in history if turn]
    if any(_REFUSAL.search(_fold(turn)) for turn in user_turns):
        return None
    if _UNSUITABLE.search(_fold(question)):
        return None
    if (
        config.CONTACT_EMAIL.casefold() in answer.casefold()
        or config.CALENDAR_BOOKING_URL in answer
    ):
        return None
    scope = _fold(question)
    if _FOLLOWUP.search(scope) and history:
        scope += " " + _fold(history[-1][0])
    if re.search(r"\b(?:those|these|that|this|ces|cela|ceci)\b", scope) and history:
        scope += " " + _fold(history[-1][0])
    requested_tools = set(
        re.findall(
            r"\b(?:kafka|spark|power bi|dax|ec2|s3|pub/sub|dataflow|nifi|lakeflow|snowflake|dbt|tableau|bigquery)\b",
            scope,
        )
    )
    if re.search(
        r"never used|no (?:\w+ )?experience|jamais utilis|aucune experience",
        _fold(answer),
    ):
        return None
    subjects = _requested_subjects(scope)
    sentences = re.split(r"(?<=[.!?])\s+|\n+", _fold(answer))
    if not any(
        re.search(requested, scope)
        and any(
            _genuine_limit(sentence)
            and re.search(topic, sentence, re.I)
            and (
                not subjects
                or any(
                    re.search(r"\b" + re.escape(subject) + r"\b", sentence)
                    for subject in subjects
                )
            )
            and (
                not requested_tools
                or any(
                    re.search(r"\b" + re.escape(tool) + r"\b", sentence)
                    for tool in requested_tools
                )
            )
            for sentence in sentences
        )
        for requested, topic in _TOPICS
    ):
        return None
    recipient = config.CONTACT_EMAIL
    if not re.fullmatch(r"[^@\s?&#]+@[^@\s?&#]+\.[^@\s?&#]+", recipient):
        return None
    try:
        calendar = urlsplit(config.CALENDAR_BOOKING_URL)
        valid_calendar = (
            calendar.scheme == "https"
            and calendar.hostname == "calendar.app.google"
            and not calendar.username
            and not calendar.password
            and calendar.port in {None, 443}
        )
    except ValueError:
        valid_calendar = False
    if not valid_calendar:
        return None
    lang = "fr" if language.lower().startswith("fr") else "en"
    copy = COPY[lang]
    return {
        "type": "contextual_handoff",
        "version": 1,
        "heading": copy["heading"],
        "description": copy["description"],
        "dismiss_text": copy["dismiss_text"],
        "links": [
            {
                "kind": "email",
                "text": copy["email"],
                "description": copy["email_hint"],
                "url": _email_draft_url(question, lang, recipient),
            },
            {
                "kind": "booking",
                "text": copy["booking"],
                "description": copy["booking_hint"],
                "url": config.CALENDAR_BOOKING_URL,
            },
        ],
    }
