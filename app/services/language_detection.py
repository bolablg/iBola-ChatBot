"""
Language detection service for automatic welcome messages.
"""

import json
import os
from typing import Dict, List


class LanguageDetection:
    """Service for detecting user language and providing localized messages."""

    def __init__(self):
        self.supported_languages = {
            "en": "English",
            "fr": "Français",
            "es": "Español",
            "de": "Deutsch",
            "it": "Italiano",
            "pt": "Português",
            "ru": "Русский",
            "zh": "中文",
            "ja": "日本語",
            "ko": "한국어",
        }

        self.welcome_messages = {
            "en": [
                "Hello! I'm iBola, your AI assistant for Bolaji's professional journey.",
                "Welcome! I'm here to help you learn about Bolaji's professional experience."
            ],
            "fr": [
                "Bonjour ! Je suis iBola, votre assistant IA pour le parcours professionnel de Bolaji.",
                "Bienvenue ! Je suis là pour vous aider à découvrir l'expérience professionnelle de Bolaji."
            ],
            "es": [
                "¡Hola! Soy iBola, tu asistente de IA para la trayectoria profesional de Bolaji.",
                "¡Bienvenido! Estoy aquí para ayudarte a conocer la experiencia profesional de Bolaji."
            ],
            "de": [
                "Hallo! Ich bin iBola, Ihr KI-Assistent für Bolajis berufliche Laufbahn.",
                "Willkommen! Ich bin hier, um Ihnen bei der Entdeckung von Bolajis Berufserfahrung zu helfen."
            ],
            "it": [
                "Ciao! Sono iBola, il tuo assistente IA per il percorso professionale di Bolaji.",
                "Benvenuto! Sono qui per aiutarti a scoprire l'esperienza professionale di Bolaji."
            ],
            "pt": [
                "Olá! Sou iBola, seu assistente de IA para a trajetória profissional de Bolaji.",
                "Bem-vindo! Estou aqui para ajudá-lo a conhecer a experiência profissional de Bolaji."
            ],
            "ru": [
                "Привет! Я iBola, ваш ИИ-ассистент для профессионального пути Боладжи.",
                "Добро пожаловать! Я здесь, чтобы помочь вам узнать о профессиональном опыте Боладжи."
            ],
            "zh": [
                "你好！我是iBola，Bolaji职业生涯的AI助手。",
                "欢迎！我是来帮助您了解Bolaji专业经验的。"
            ],
            "ja": [
                "こんにちは！私はiBola、Bolajiの職業経歴のAIアシスタントです。",
                "ようこそ！私はBolajiの職業経験について知っていただくお手伝いをいたします。"
            ],
            "ko": [
                "안녕하세요! 저는 iBola, Bolaji의 직업적 여정을 위한 AI 어시스턴트입니다.",
                "환영합니다! 저는 Bolaji의 직업적 경험에 대해 알려드리기 위해 여기 있습니다."
            ],
        }

        self.agent_translations = {
            "en": {
                "professional": "Professional Expert",
                "education": "Education Specialist",
                "learning": "Learning Advisor",
                "redirect": "iBola Assistant",
            },
            "fr": {
                "professional": "Expert Professionnel",
                "education": "Spécialiste Éducation",
                "learning": "Conseiller Apprentissage",
                "redirect": "Assistant iBola",
            },
            "es": {
                "professional": "Experto Profesional",
                "education": "Especialista en Educación",
                "learning": "Asesor de Aprendizaje",
                "redirect": "Asistente iBola",
            },
            "de": {
                "professional": "Fachmann",
                "education": "Bildungsspezialist",
                "learning": "Lernberater",
                "redirect": "iBola-Assistent",
            },
            "it": {
                "professional": "Esperto Professionale",
                "education": "Specialista Istruzione",
                "learning": "Consulente Apprendimento",
                "redirect": "Assistente iBola",
            },
            "pt": {
                "professional": "Especialista Profissional",
                "education": "Especialista em Educação",
                "learning": "Consultor de Aprendizado",
                "redirect": "Assistente iBola",
            },
            "ru": {
                "professional": "Профессиональный эксперт",
                "education": "Специалист по образованию",
                "learning": "Консультант по обучению",
                "redirect": "Ассистент iBola",
            },
            "zh": {
                "professional": "专业专家",
                "education": "教育专家",
                "learning": "学习顾问",
                "redirect": "iBola助手",
            },
            "ja": {
                "professional": "専門家",
                "education": "教育専門家",
                "learning": "学習アドバイザー",
                "redirect": "iBolaアシスタント",
            },
            "ko": {
                "professional": "전문가",
                "education": "교육 전문가",
                "learning": "학습 상담사",
                "redirect": "iBola 어시스턴트",
            },
        }

    def detect_language(self, browser_language: str = None, message: str = None) -> str:
        """
        Detect the user's preferred language.

        Args:
            browser_language: Browser's navigator.language
            message: Optional message text for language detection

        Returns:
            str: Language code (e.g., 'en', 'fr', etc.)
        """
        # Priority 1: Browser language
        if browser_language:
            primary_lang = browser_language.split("-")[0].lower()
            if primary_lang in self.supported_languages:
                return primary_lang

        # Priority 2: Message content analysis (if provided)
        if message:
            detected_lang = self._analyze_message_language(message)
            if detected_lang:
                return detected_lang

        # Default to English
        return "en"

    def _analyze_message_language(self, message: str) -> str:
        """Analyze message content to detect language."""
        import re

        # Simple language detection based on common words/phrases
        message_lower = message.lower()

        # Spanish indicators (check first to avoid conflicts with French)
        spanish_patterns = [
            r'\bel\b', r'\bla\b', r'\blos\b', r'\blas\b', r'\by\b', r'\bes\b',
            r'\bhola\b', r'\bsoy\b', r'\beres\b', r'\bsomos\b', r'\bgracias\b'
        ]
        if any(re.search(pattern, message_lower) for pattern in spanish_patterns):
            return "es"

        # French indicators
        french_patterns = [
            r'\ble\b', r'\bla\b', r'\bles\b', r'\bet\b', r'\best\b',
            r'\bje\b', r'\btu\b', r'\bnous\b', r'\bvous\b', r'\bsuis\b', r'\bêtes\b'
        ]
        if any(re.search(pattern, message_lower) for pattern in french_patterns):
            return "fr"

        # German indicators
        german_patterns = [
            r'\bder\b', r'\bdie\b', r'\bdas\b', r'\bund\b', r'\bist\b',
            r'\bich\b', r'\bdu\b', r'\bwir\b', r'\bihr\b'
        ]
        if any(re.search(pattern, message_lower) for pattern in german_patterns):
            return "de"

        # Italian indicators
        italian_patterns = [
            r'\bil\b', r'\blo\b', r'\bla\b', r'\bi\b', r'\bgli\b', r'\ble\b',
            r'\be\b', r'\bè\b', r'\bsono\b', r'\bciao\b'
        ]
        if any(re.search(pattern, message_lower) for pattern in italian_patterns):
            return "it"

        # Portuguese indicators
        portuguese_patterns = [
            r'\bo\b', r'\ba\b', r'\bos\b', r'\bas\b', r'\be\b', r'\bé\b',
            r'\bsou\b', r'\bés\b', r'\bsomos\b', r'\bolá\b'
        ]
        if any(re.search(pattern, message_lower) for pattern in portuguese_patterns):
            return "pt"

        return None

    def get_welcome_messages(self, language: str) -> List[str]:
        """Get welcome messages in the specified language."""
        return self.welcome_messages.get(language, self.welcome_messages["en"])

    def get_agent_name(self, agent_type: str, language: str) -> str:
        """Get localized agent name."""
        lang_translations = self.agent_translations.get(
            language, self.agent_translations["en"]
        )
        return lang_translations.get(agent_type, lang_translations["redirect"])

    def get_localized_text(self, key: str, language: str) -> str:
        """Get localized text for various UI elements."""
        localized_texts = {
            "en": {
                "readjust_question": "🔄 Readjust Question",
                "professional_exp": "💼 Professional Experience",
                "education_bg": "🎓 Education Background",
                "learning_advice": "📚 Learning Advice",
                "book_appointment": "📅 Book Appointment",
                "send_email": "📧 Send Email",
                "end_chat": "👋 End Chat",
                "help_rephrase": "Help me rephrase my question to get relevant information",
                "ask_career": "Ask about career journey and achievements",
                "learn_academic": "Learn about academic qualifications",
                "guidance_skills": "Get guidance on learning data science & AI skills",
                "schedule_meeting": "Schedule a meeting with Bolaji",
                "send_email_bolaji": "Send an email to Bolaji",
                "thank_interest": "Thank you for your interest",
            },
            "fr": {
                "readjust_question": "🔄 Reformuler la Question",
                "professional_exp": "💼 Expérience Professionnelle",
                "education_bg": "🎓 Formation Académique",
                "learning_advice": "📚 Conseils d'Apprentissage",
                "book_appointment": "📅 Prendre Rendez-vous",
                "send_email": "📧 Envoyer un Email",
                "end_chat": "👋 Terminer la Discussion",
                "help_rephrase": "Aidez-moi à reformuler ma question pour obtenir des informations pertinentes",
                "ask_career": "Demander sur le parcours professionnel et les réalisations",
                "learn_academic": "En savoir plus sur les qualifications académiques",
                "guidance_skills": "Obtenir des conseils sur l'apprentissage des compétences en data science et IA",
                "schedule_meeting": "Planifier une réunion avec Bolaji",
                "send_email_bolaji": "Envoyer un email à Bolaji",
                "thank_interest": "Merci pour votre intérêt",
            },
            "es": {
                "readjust_question": "🔄 Reformular Pregunta",
                "professional_exp": "💼 Experiencia Profesional",
                "education_bg": "🎓 Antecedentes Educativos",
                "learning_advice": "📚 Consejos de Aprendizaje",
                "book_appointment": "📅 Reservar Cita",
                "send_email": "📧 Enviar Email",
                "end_chat": "👋 Terminar Chat",
                "help_rephrase": "Ayúdame a reformular mi pregunta para obtener información relevante",
                "ask_career": "Preguntar sobre trayectoria profesional y logros",
                "learn_academic": "Aprender sobre calificaciones académicas",
                "guidance_skills": "Obtener orientación sobre aprendizaje de ciencia de datos e IA",
                "schedule_meeting": "Programar una reunión con Bolaji",
                "send_email_bolaji": "Enviar un email a Bolaji",
                "thank_interest": "Gracias por tu interés",
            },
        }

        lang_texts = localized_texts.get(language, localized_texts["en"])
        return lang_texts.get(key, localized_texts["en"].get(key, key))


# Global instance
language_service = LanguageDetection()
