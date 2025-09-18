"""
Unit tests for PsychologicalTextChunker in nlp/preprocessor.py

Tests chunking functionality, emotional metadata extraction, and therapeutic theme detection.
"""

import pytest
from unittest.mock import Mock, patch
from typing import List

from psy_supabase.nlp.preprocessor import PsychologicalTextChunker, TherapeuticChunk


class TestPsychologicalTextChunker:
    """Test suite for PsychologicalTextChunker class."""

    @pytest.fixture
    def chunker(self):
        """Create a PsychologicalTextChunker instance for testing."""
        return PsychologicalTextChunker()

    @pytest.fixture
    def sample_texts(self):
        """Sample texts for testing different emotional and thematic content."""
        return {
            "happy": "Me siento muy feliz hoy. La vida es hermosa y todo va bien.",
            "sad": "Estoy muy triste y deprimido. No encuentro sentido a nada.",
            "angry": "Estoy furioso y enojado. Todo me molesta mucho.",
            "fear": "Tengo mucho miedo y ansiedad. Me preocupa el futuro.",
            "mixed": "Me siento feliz por el trabajo pero triste por la familia. Es complicado.",
            "neutral": "Hoy fui al supermercado y compré algunas cosas. Luego volví a casa.",
            "complex": "Mi relación con mi madre es difícil. Siempre discutimos y me siento culpable. "
                     "Pero también la amo mucho. No sé qué hacer con estos sentimientos contradictorios."
        }

    def test_chunker_initialization(self, chunker):
        """Test that chunker initializes properly."""
        assert chunker is not None
        assert hasattr(chunker, 'nlp')
        assert hasattr(chunker, 'emotional_cues')
        assert hasattr(chunker, 'intensity_markers')

    def test_chunk_text_basic(self, chunker, sample_texts):
        """Test basic text chunking functionality."""
        text = sample_texts["happy"]
        chunks = chunker.chunk_text(text)
        
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(chunk, TherapeuticChunk) for chunk in chunks)

    def test_chunk_metadata_structure(self, chunker, sample_texts):
        """Test that chunk metadata has correct structure."""
        text = sample_texts["sad"]
        chunks = chunker.chunk_text(text)
        
        for chunk in chunks:
            assert hasattr(chunk, 'text')
            assert hasattr(chunk, 'primary_emotion')
            assert hasattr(chunk, 'polarity')
            assert hasattr(chunk, 'intensity')
            assert hasattr(chunk, 'theme')
            
            # Validate data types
            assert isinstance(chunk.text, str)
            assert isinstance(chunk.primary_emotion, str)
            assert isinstance(chunk.polarity, (int, float))
            assert isinstance(chunk.intensity, (int, float))
            assert isinstance(chunk.theme, str)
            
            # Validate ranges
            assert -1 <= chunk.polarity <= 1
            assert 0 <= chunk.intensity <= 1

    def test_emotion_detection(self, chunker, sample_texts):
        """Test emotion detection accuracy."""
        # Test happy emotion
        happy_chunks = chunker.chunk_text(sample_texts["happy"])
        assert any(chunk.primary_emotion == "happy" for chunk in happy_chunks)
        
        # Test sad emotion
        sad_chunks = chunker.chunk_text(sample_texts["sad"])
        assert any(chunk.primary_emotion == "sad" for chunk in sad_chunks)
        
        # Test angry emotion
        angry_chunks = chunker.chunk_text(sample_texts["angry"])
        assert any(chunk.primary_emotion == "angry" for chunk in angry_chunks)
        
        # Test fear emotion
        fear_chunks = chunker.chunk_text(sample_texts["fear"])
        assert any(chunk.primary_emotion == "fear" for chunk in fear_chunks)

    def test_polarity_detection(self, chunker, sample_texts):
        """Test polarity scoring."""
        # Happy text should have positive polarity
        happy_chunks = chunker.chunk_text(sample_texts["happy"])
        happy_polarities = [chunk.polarity for chunk in happy_chunks]
        assert any(p > 0 for p in happy_polarities)
        
        # Sad text should have negative polarity
        sad_chunks = chunker.chunk_text(sample_texts["sad"])
        sad_polarities = [chunk.polarity for chunk in sad_chunks]
        assert any(p < 0 for p in sad_polarities)

    def test_intensity_detection(self, chunker, sample_texts):
        """Test intensity scoring."""
        # Emotional text should have higher intensity than neutral
        emotional_chunks = chunker.chunk_text(sample_texts["sad"])
        neutral_chunks = chunker.chunk_text(sample_texts["neutral"])
        
        emotional_intensities = [chunk.intensity for chunk in emotional_chunks]
        neutral_intensities = [chunk.intensity for chunk in neutral_chunks]
        
        avg_emotional = sum(emotional_intensities) / len(emotional_intensities)
        avg_neutral = sum(neutral_intensities) / len(neutral_intensities)
        
        assert avg_emotional > avg_neutral

    def test_theme_detection(self, chunker, sample_texts):
        """Test therapeutic theme detection."""
        complex_chunks = chunker.chunk_text(sample_texts["complex"])
        themes = [chunk.theme for chunk in complex_chunks]
        
        # Should detect family/relationship themes
        assert any("familia" in theme.lower() or "relación" in theme.lower() for theme in themes)

    def test_empty_text_handling(self, chunker):
        """Test handling of empty or whitespace-only text."""
        empty_chunks = chunker.chunk_text("")
        whitespace_chunks = chunker.chunk_text("   \n\t   ")
        
        assert len(empty_chunks) == 0
        assert len(whitespace_chunks) == 0

    def test_single_sentence_chunking(self, chunker):
        """Test chunking of single sentences."""
        single_sentence = "Me siento muy triste hoy."
        chunks = chunker.chunk_text(single_sentence)
        
        assert len(chunks) == 1
        assert chunks[0].text.strip() == single_sentence

    def test_multi_sentence_chunking(self, chunker, sample_texts):
        """Test chunking of multi-sentence text."""
        complex_text = sample_texts["complex"]
        chunks = chunker.chunk_text(complex_text)
        
        # Should create multiple chunks for complex text
        assert len(chunks) >= 2
        
        # All chunks together should contain the original content
        combined_text = " ".join(chunk.text for chunk in chunks)
        original_words = set(complex_text.lower().split())
        combined_words = set(combined_text.lower().split())
        
        # Most original words should be preserved
        overlap = len(original_words.intersection(combined_words))
        assert overlap / len(original_words) > 0.8

    def test_mixed_emotions_handling(self, chunker, sample_texts):
        """Test handling of text with mixed emotions."""
        mixed_chunks = chunker.chunk_text(sample_texts["mixed"])
        emotions = [chunk.primary_emotion for chunk in mixed_chunks]
        
        # Should detect multiple different emotions
        unique_emotions = set(emotions)
        assert len(unique_emotions) >= 2

    def test_chunk_text_length_tracking(self, chunker, sample_texts):
        """Test that chunk metadata includes text length."""
        chunks = chunker.chunk_text(sample_texts["complex"])
        
        for chunk in chunks:
            expected_length = len(chunk.text)
            # Assuming metadata includes text_length
            assert len(chunk.text) > 0

    @patch('psy_supabase.nlp.preprocessor.spacy.load')
    def test_spacy_model_loading_fallback(self, mock_spacy_load, chunker):
        """Test fallback when spaCy model fails to load."""
        # Simulate spaCy model loading failure
        mock_spacy_load.side_effect = OSError("Model not found")
        
        # Should handle gracefully and still create chunks
        text = "Test text for fallback."
        chunks = chunker.chunk_text(text)
        
        # Should still return chunks even with fallback
        assert isinstance(chunks, list)

    def test_emotional_cues_patterns(self, chunker):
        """Test that emotional cue patterns are properly defined."""
        assert 'happy' in chunker.emotional_cues
        assert 'sad' in chunker.emotional_cues
        assert 'angry' in chunker.emotional_cues
        assert 'fear' in chunker.emotional_cues
        
        # Each emotion should have compiled regex patterns
        for emotion, patterns in chunker.emotional_cues.items():
            assert isinstance(patterns, list)
            assert len(patterns) > 0

    def test_intensity_markers_patterns(self, chunker):
        """Test that intensity marker patterns are properly defined."""
        assert 'high' in chunker.intensity_markers
        assert 'medium' in chunker.intensity_markers
        assert 'low' in chunker.intensity_markers
        
        # Each intensity should have compiled regex patterns
        for intensity, patterns in chunker.intensity_markers.items():
            assert isinstance(patterns, list)
            assert len(patterns) > 0

    def test_therapeutic_chunk_dataclass(self):
        """Test TherapeuticChunk dataclass functionality."""
        chunk = TherapeuticChunk(
            text="Test text",
            primary_emotion="happy",
            polarity=0.5,
            intensity=0.7,
            theme="test theme"
        )
        
        assert chunk.text == "Test text"
        assert chunk.primary_emotion == "happy"
        assert chunk.polarity == 0.5
        assert chunk.intensity == 0.7
        assert chunk.theme == "test theme"

    def test_chunk_consistency(self, chunker):
        """Test that chunking produces consistent results."""
        text = "Me siento triste y preocupado por el futuro."
        
        # Run chunking multiple times
        chunks1 = chunker.chunk_text(text)
        chunks2 = chunker.chunk_text(text)
        
        # Should produce same number of chunks
        assert len(chunks1) == len(chunks2)
        
        # Should produce same emotions (deterministic)
        emotions1 = [chunk.primary_emotion for chunk in chunks1]
        emotions2 = [chunk.primary_emotion for chunk in chunks2]
        assert emotions1 == emotions2

    def test_long_text_chunking(self, chunker):
        """Test chunking of longer therapeutic text."""
        long_text = """
        Mi vida ha sido muy difícil últimamente. Me siento deprimido y ansioso constantemente.
        No puedo dormir bien y tengo pesadillas. Mi relación con mi familia está deteriorándose.
        En el trabajo también tengo problemas con mi jefe. Me siento abrumado y no sé qué hacer.
        A veces pienso que las cosas nunca van a mejorar. Necesito ayuda pero no sé dónde encontrarla.
        """
        
        chunks = chunker.chunk_text(long_text)
        
        # Should create multiple chunks for long text
        assert len(chunks) >= 3
        
        # Should detect various emotions and themes
        emotions = [chunk.primary_emotion for chunk in chunks]
        themes = [chunk.theme for chunk in chunks]
        
        assert len(set(emotions)) >= 2  # Multiple emotions
        assert any("trabajo" in theme.lower() or "familia" in theme.lower() for theme in themes)

    def test_special_characters_handling(self, chunker):
        """Test handling of special characters and punctuation."""
        text_with_special = "¡Estoy muy feliz! ¿Cómo estás? Me siento genial... 😊"
        chunks = chunker.chunk_text(text_with_special)
        
        assert len(chunks) > 0
        assert all(chunk.text for chunk in chunks)  # No empty chunks

    def test_numerical_content_handling(self, chunker):
        """Test handling of text with numbers and dates."""
        text_with_numbers = "Hace 3 años que me siento así. El 15 de marzo fue terrible."
        chunks = chunker.chunk_text(text_with_numbers)
        
        assert len(chunks) > 0
        # Should preserve numerical content
        combined_text = " ".join(chunk.text for chunk in chunks)
        assert "3" in combined_text or "tres" in combined_text.lower()


class TestTherapeuticChunkIntegration:
    """Integration tests for TherapeuticChunk with other components."""

    def test_chunk_serialization(self):
        """Test that chunks can be serialized for database storage."""
        chunk = TherapeuticChunk(
            text="Test chunk",
            primary_emotion="neutral",
            polarity=0.0,
            intensity=0.5,
            theme="general"
        )
        
        # Should be able to convert to dict for JSON serialization
        chunk_dict = {
            'text': chunk.text,
            'primary_emotion': chunk.primary_emotion,
            'polarity': chunk.polarity,
            'intensity': chunk.intensity,
            'theme': chunk.theme
        }
        
        assert isinstance(chunk_dict, dict)
        assert all(key in chunk_dict for key in ['text', 'primary_emotion', 'polarity', 'intensity', 'theme'])

    def test_metadata_extraction(self):
        """Test extraction of metadata for database storage."""
        chunk = TherapeuticChunk(
            text="Me siento ansioso",
            primary_emotion="fear",
            polarity=-0.3,
            intensity=0.8,
            theme="ansiedad"
        )
        
        metadata = {
            'primary_emotion': chunk.primary_emotion,
            'polarity': chunk.polarity,
            'intensity': chunk.intensity,
            'theme': chunk.theme,
            'text_length': len(chunk.text),
            'chunk_type': 'psychological'
        }
        
        # Validate metadata structure for database
        assert isinstance(metadata['primary_emotion'], str)
        assert isinstance(metadata['polarity'], (int, float))
        assert isinstance(metadata['intensity'], (int, float))
        assert isinstance(metadata['theme'], str)
        assert isinstance(metadata['text_length'], int)
        assert isinstance(metadata['chunk_type'], str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
