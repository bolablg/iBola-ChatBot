"""
Unit tests for service components.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, mock_open

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.language_detection import LanguageDetection, language_service
from app.services.dynamic_guardrails import DynamicGuardrails, dynamic_guardrails


class TestLanguageDetection:
    """Test cases for Language Detection service."""

    def test_supported_languages(self):
        """Test supported languages list."""
        service = LanguageDetection()
        assert 'en' in service.supported_languages
        assert 'fr' in service.supported_languages
        assert 'es' in service.supported_languages
        assert len(service.supported_languages) >= 10

    def test_language_detection(self):
        """Test language detection from browser language."""
        service = LanguageDetection()

        # Test exact matches
        assert service.detect_language('en-US') == 'en'
        assert service.detect_language('fr-FR') == 'fr'
        assert service.detect_language('es-ES') == 'es'

        # Test unsupported language defaults to English
        assert service.detect_language('xx-XX') == 'en'

        # Test None input
        assert service.detect_language(None) == 'en'

    def test_message_language_analysis(self):
        """Test language detection from message content."""
        service = LanguageDetection()

        # Test French detection
        assert service._analyze_message_language("Bonjour, comment allez-vous?") == 'fr'
        assert service._analyze_message_language("Je suis développeur") == 'fr'

        # Test Spanish detection
        assert service._analyze_message_language("Hola, ¿cómo estás?") == 'es'
        assert service._analyze_message_language("Soy programador") == 'es'

        # Test unknown language
        assert service._analyze_message_language("Hello world") is None

    def test_welcome_messages(self):
        """Test welcome message generation."""
        service = LanguageDetection()

        # Test English messages
        en_messages = service.get_welcome_messages('en')
        assert len(en_messages) == 2
        assert "Hello" in en_messages[0]

        # Test French messages
        fr_messages = service.get_welcome_messages('fr')
        assert len(fr_messages) == 2
        assert "Bonjour" in fr_messages[0]

        # Test unsupported language defaults to English
        default_messages = service.get_welcome_messages('xx')
        assert default_messages == service.get_welcome_messages('en')

    def test_agent_translations(self):
        """Test agent name translations."""
        service = LanguageDetection()

        # Test English
        assert service.get_agent_name('professional', 'en') == 'Professional Expert'

        # Test French
        assert service.get_agent_name('professional', 'fr') == 'Expert Professionnel'

        # Test unknown agent type
        assert service.get_agent_name('unknown', 'en') == 'iBola Assistant'


class TestDynamicGuardrails:
    """Test cases for Dynamic Guardrails service."""

    def test_initialization(self):
        """Test guardrails initialization."""
        guardrails = DynamicGuardrails()
        assert hasattr(guardrails, 'professional_keywords')
        assert hasattr(guardrails, 'education_keywords')
        assert hasattr(guardrails, 'learning_keywords')
        assert hasattr(guardrails, 'off_topic_indicators')
        assert len(guardrails.professional_keywords) > 0

    def test_message_analysis(self):
        """Test message analysis for categorization."""
        guardrails = DynamicGuardrails()

        # Test professional query
        analysis = guardrails.analyze_message("What projects have you worked on?")
        assert analysis['primary_category'] in ['professional', 'learning']
        assert 'confidence' in analysis
        assert 'scores' in analysis

        # Test education query
        analysis = guardrails.analyze_message("Where did you study?")
        assert analysis['primary_category'] in ['education', 'learning']
        assert 'confidence' in analysis

        # Test off-topic query
        analysis = guardrails.analyze_message("What's the weather like?")
        assert analysis['primary_category'] == 'off_topic'

    def test_context_analysis(self):
        """Test analysis with conversation context."""
        guardrails = DynamicGuardrails()

        context = [
            "What is your background?",
            "I have experience in data science.",
            "Tell me more about your education."
        ]

        analysis = guardrails.analyze_message("What university did you attend?", context)
        assert analysis['primary_category'] == 'education'

    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_feedback_learning(self, mock_json_dump, mock_file):
        """Test feedback learning mechanism."""
        guardrails = DynamicGuardrails()

        # Test learning from feedback
        initial_prof_count = len(guardrails.professional_keywords)

        guardrails.learn_from_feedback(
            "I worked on machine learning projects",
            "professional",
            ["Tell me about your work experience"]
        )

        # Check that keywords were added
        assert len(guardrails.professional_keywords) >= initial_prof_count
        assert mock_json_dump.called  # Verify data was saved

    def test_statistics_generation(self):
        """Test statistics generation."""
        guardrails = DynamicGuardrails()

        stats = guardrails.get_statistics()
        assert 'total_feedback_entries' in stats
        assert 'category_distribution' in stats
        assert 'keyword_counts' in stats
        assert 'last_updated' in stats

        # Check keyword counts
        assert 'professional' in stats['keyword_counts']
        assert 'education' in stats['keyword_counts']
        assert 'learning' in stats['keyword_counts']
        assert 'off_topic' in stats['keyword_counts']


class TestGlobalInstances:
    """Test global service instances."""

    def test_language_service_instance(self):
        """Test global language service instance."""
        assert isinstance(language_service, LanguageDetection)
        assert hasattr(language_service, 'supported_languages')
        assert hasattr(language_service, 'get_welcome_messages')

    def test_dynamic_guardrails_instance(self):
        """Test global dynamic guardrails instance."""
        assert isinstance(dynamic_guardrails, DynamicGuardrails)
        assert hasattr(dynamic_guardrails, 'professional_keywords')
        assert hasattr(dynamic_guardrails, 'analyze_message')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
