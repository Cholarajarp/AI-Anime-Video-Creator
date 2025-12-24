"""
Translation Service for Multi-Language Support
Translates prompts and text to selected language
"""

from typing import Optional
from loguru import logger


class TranslationService:
    """Handle text translation for multi-language support."""

    def __init__(self):
        self.language_codes = {
            "Japanese": "ja",
            "English": "en",
            "Hindi": "hi",
            "Kannada": "kn",
            "Korean": "ko",
            "Chinese": "zh-CN",
            "Tamil": "ta",
            "Telugu": "te",
            "Spanish": "es",
            "French": "fr",
            "German": "de",
        }

    def translate_text(
        self,
        text: str,
        target_language: str,
        source_language: str = "auto"
    ) -> str:
        """
        Translate text to target language.
        Uses deep-translator library for free translation.
        """

        if not text or not text.strip():
            return text

        try:
            # Try using deep-translator (more compatible)
            try:
                from deep_translator import GoogleTranslator

                target_code = self.language_codes.get(target_language, target_language)

                logger.info(f"🌍 Translating to {target_language}...")

                translator = GoogleTranslator(source='auto', target=target_code)
                translated = translator.translate(text)

                logger.info(f"✅ Translation complete: {translated[:50]}...")

                return translated

            except ImportError:
                # Install deep-translator
                logger.info("Installing deep-translator...")
                import subprocess
                import sys
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "deep-translator"
                ])

                # Retry
                return self.translate_text(text, target_language, source_language)

        except Exception as e:
            logger.warning(f"Translation failed: {e}, using original text")
            return text

    def get_language_from_voice(self, voice_id: str) -> str:
        """Extract language from voice ID."""

        voice_map = {
            "ja-JP": "Japanese",
            "en-US": "English",
            "hi-IN": "Hindi",
            "kn-IN": "Kannada",
            "ko-KR": "Korean",
            "zh-CN": "Chinese",
            "ta-IN": "Tamil",
            "te-IN": "Telugu",
        }

        for code, lang in voice_map.items():
            if code in voice_id:
                return lang

        return "English"

    def enhance_prompt_with_language(
        self,
        prompt: str,
        language: str
    ) -> str:
        """
        Enhance prompt with language-specific style keywords.
        """

        enhancements = {
            "Japanese": ", anime style, japanese animation, detailed anime art",
            "Korean": ", manhwa style, korean webtoon, vibrant colors",
            "Chinese": ", donghua style, chinese animation, elegant composition",
            "Hindi": ", bollywood style, vibrant colors, dramatic lighting",
            "Kannada": ", south indian aesthetic, rich colors, cultural elements",
        }

        enhancement = enhancements.get(language, "")

        if enhancement and enhancement not in prompt:
            return prompt + enhancement

        return prompt

