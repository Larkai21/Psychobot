"""
Psychological Text Chunker for therapeutic applications.

This module provides specialized text chunking that segments user messages
by emotional meaning and psychological themes rather than just syntax.
Each chunk includes metadata about primary emotion, polarity, intensity,
and psychological theme for enhanced therapeutic response generation.
"""

import re
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

import spacy
from spacy.tokens import Doc, Span  # ✅ CAMBIO: Sent → Span

from psy_supabase import get_package_logger
from psy_supabase.utilities.nlp_utils import get_spacy_model
from psy_supabase.utilities.therapeutic_mappings import TherapeuticMappings

logger = get_package_logger(__name__)


@dataclass
class TherapeuticChunk:
    text: str
    primary_emotion: str
    polarity: float
    intensity: float
    theme: str


class PsychologicalTextChunker:
    EMOTIONAL_CUES = {
        "high_distress": [
            (r"\b(ya no puedo|no puedo más|estoy desesperado|me quiero morir)\b", 0.9),
            (r"\b(me duele mucho|dolor insoportable|sufrimiento|agonía)\b", 0.8),
            (r"\b(odio mi vida|todo está mal|no hay salida)\b", 0.9),
        ],
        "medium_distress": [
            (r"\b(me siento|estoy triste|me duele|lo que más me duele)\b", 0.6),
            (r"\b(pero|sin embargo|aunque|a pesar de)\b", 0.5),
            (r"\b(estoy cansado de|ya no sé|no entiendo)\b", 0.6),
        ],
        "low_distress": [
            (r"\b(un poco|algo|quizás|tal vez|creo que)\b", 0.3),
            (r"\b(me preocupa|me inquieta|me molesta)\b", 0.4),
        ],
        "positive": [
            (r"\b(me siento bien|estoy feliz|me alegra|es bueno)\b", 0.7),
            (r"\b(gracias|agradezco|me ayuda|es útil)\b", 0.6),
        ]
    }

    EMOTION_PATTERNS = {
        "sad": [
            r"\b(triste|tristeza|melancol|deprim|llor|pena|dolor emocional)\b",
            r"\b(vacío|vacía|sin sentido|desesperanza|desaliento)\b",
        ],
        "angry": [
            r"\b(enojado|enojada|furioso|furiosa|rabia|ira|molesto|molesta)\b",
            r"\b(odio|detesto|me irrita|me enfurece|indignado|indignada)\b",
        ],
        "fear": [
            r"\b(miedo|temor|terror|pánico|ansiedad|nervioso|nerviosa)\b",
            r"\b(preocup|inquiet|angust|asust|espant)\b",
        ],
        "happy": [
            r"\b(feliz|alegr|content|satisfech|bien|mejor)\b",
            r"\b(esperanza|optimis|positiv|anim)\b",
        ],
        "neutral": [
            r"\b(normal|regular|así|igual|común)\b",
        ]
    }

    def __init__(self, model_name: str = "es_core_news_sm"):
        self.model_name = model_name
        self.nlp = get_spacy_model(model_name)
        if not self.nlp:
            logger.error(f"Failed to load spaCy model: {model_name}")
            raise RuntimeError(f"Could not load spaCy model: {model_name}")
        self.therapeutic_mappings = TherapeuticMappings()
        logger.info(f"Initialized PsychologicalTextChunker with model: {model_name}")

    def chunk_text(self, text: str) -> List[TherapeuticChunk]:
        if not text or not text.strip():
            return []
        try:
            doc = self.nlp(text.strip())
            semantic_chunks = self._extract_semantic_chunks(doc)
            return [self._create_therapeutic_chunk(chunk) for chunk in semantic_chunks]
        except Exception as e:
            logger.error(f"Error chunking text: {e}")
            return [TherapeuticChunk(
                text=text.strip(),
                primary_emotion="neutral",
                polarity=0.0,
                intensity=0.5,
                theme="general_support"
            )]

    def _extract_semantic_chunks(self, doc: Doc) -> List[str]:
        sentences = list(doc.sents)  # ✅ CAMBIO: convertir a lista
        if not sentences:
            return [doc.text]

        chunks, current_chunk, current_emotion = [], [], None

        for sent in sentences:
            sent_text = sent.text.strip()
            if not sent_text:
                continue
            sent_emotion = self._detect_primary_emotion(sent_text)
            if self._should_start_new_chunk(current_emotion, sent_emotion, current_chunk):
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                current_emotion = sent_emotion
            current_chunk.append(sent_text)

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks if chunks else [doc.text]

    def _should_start_new_chunk(self, current_emotion: Optional[str], new_emotion: str, current_chunk: List[str]) -> bool:
        if current_emotion is None:
            return False
        if current_emotion != new_emotion and new_emotion != "neutral":
            return True
        if len(current_chunk) >= 3:
            return True
        return False

    def _create_therapeutic_chunk(self, text: str) -> TherapeuticChunk:
        primary_emotion = self._detect_primary_emotion(text)
        polarity = self._calculate_polarity(text, primary_emotion)
        intensity = self._calculate_intensity(text)
        theme = self._detect_psychological_theme(text)
        return TherapeuticChunk(text, primary_emotion, polarity, intensity, theme)

    def _detect_primary_emotion(self, text: str) -> str:
        text_lower = text.lower()
        emotion_scores = {emotion: sum(len(re.findall(p, text_lower)) for p in patterns)
                          for emotion, patterns in self.EMOTION_PATTERNS.items()}
        return max(emotion_scores, key=emotion_scores.get) if max(emotion_scores.values()) > 0 else "neutral"

    def _calculate_polarity(self, text: str, primary_emotion: str) -> float:
        base = {"happy": 0.7, "sad": -0.6, "angry": -0.7, "fear": -0.5, "neutral": 0.0}.get(primary_emotion, 0.0)
        adjustment, text_lower = 0.0, text.lower()
        for cue_type, patterns in self.EMOTIONAL_CUES.items():
            for pattern, weight in patterns:
                if re.search(pattern, text_lower):
                    adjustment += weight * 0.3 if cue_type == "positive" else -weight * 0.2
        return round(max(-1.0, min(1.0, base + adjustment)), 2)

    def _calculate_intensity(self, text: str) -> float:
        text_lower, intensity = text.lower(), 0.3
        for cue_type, patterns in self.EMOTIONAL_CUES.items():
            for pattern, weight in patterns:
                if re.search(pattern, text_lower):
                    intensity = max(intensity, weight)
        for pattern, weight in [
            (r"\b(muy|mucho|demasiado|extremadamente)\b", 0.3),
            (r"(!!|¡¡|!!!)", 0.2),
            (r"[A-Z]{3,}", 0.2),
            (r"\b(siempre|nunca|todo|nada|completamente)\b", 0.2),
        ]:
            if re.search(pattern, text_lower):
                intensity += weight
        return round(min(1.0, intensity), 2)

    def _detect_psychological_theme(self, text: str) -> str:
        text_lower, theme_scores = text.lower(), {}
        for theme, keywords in self.therapeutic_mappings.ENHANCED_TAXONOMY.items():
            score = sum(1 for keyword in keywords if keyword.lower() in text_lower)
            if score: theme_scores[theme] = score
        for theme, data in self.therapeutic_mappings.THERAPEUTIC_THEMES.items():
            score = sum(2 for keyword in data.get("keywords", []) if keyword.lower() in text_lower)
            if score: theme_scores[theme] = theme_scores.get(theme, 0) + score
        return max(theme_scores, key=theme_scores.get) if theme_scores else "general_support"

    def get_chunk_summary(self, chunks: List[TherapeuticChunk]) -> Dict[str, Any]:
        if not chunks:
            return {}
        emotions = [c.primary_emotion for c in chunks]
        themes = [c.theme for c in chunks]
        return {
            "total_chunks": len(chunks),
            "dominant_emotion": max(set(emotions), key=emotions.count),
            "dominant_theme": max(set(themes), key=themes.count),
            "average_polarity": round(sum(c.polarity for c in chunks)/len(chunks), 2),
            "average_intensity": round(sum(c.intensity for c in chunks)/len(chunks), 2),
            "emotion_distribution": {e: emotions.count(e) for e in set(emotions)},
            "theme_distribution": {t: themes.count(t) for t in set(themes)},
        }