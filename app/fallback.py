FALLBACK_MESSAGES = {
    "en": {
        "answer": "I don't have specific information on that topic. For more details, please feel free to contact Bolaji directly.",
        "actions": [
            {
                "text": "📧 Email",
                "url": "mailto:hello@bolablg.com",
                "type": "link"
            },
            {
                "text": "📅 Appointment",
                "url": "https://calendar.google.com/calendar/appointments/schedules/AcZssZ3YeidR5Og4YSGZIlxUIlDAf0AiRA6N8-MAzr-Sy55BtbKhBLXkfa8M_P_92eokXRnayLVlEXiW?gv=true",
                "type": "popup"
            },
            {
                "text": "[in] LinkedIn",
                "url": "https://linkedin.com/in/bolablg",
                "type": "link"
            }
        ]
    },
    "fr": {
        "answer": "Je n'ai pas d'information spécifique sur ce sujet. Pour plus de détails, n'hésitez pas à contacter Bolaji directement.",
        "actions": [
            {
                "text": "📧 E-mail Direct",
                "url": "mailto:hello@bolablg.com",
                "type": "link"
            },
            {
                "text": "📅 Prendre rendez-vous",
                "url": "https://calendar.google.com/calendar/appointments/schedules/AcZssZ3YeidR5Og4YSGZIlxUIlDAf0AiRA6N8-MAzr-Sy55BtbKhBLXkfa8M_P_92eokXRnayLVlEXiW?gv=true",
                "type": "popup"
            },
            {
                "text": "[in] LinkedIn",
                "url": "https://linkedin.com/in/bolablg",
                "type": "link"
            }
        ]
    }
}

def fallback_response(language='en'):
    """Generate a language-specific fallback response when the chatbot cannot answer."""
    return FALLBACK_MESSAGES.get(language, FALLBACK_MESSAGES['en'])