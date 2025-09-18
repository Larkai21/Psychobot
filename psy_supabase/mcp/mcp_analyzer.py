import json
import re
import unicodedata
from pathlib import Path

# Normalización de texto
def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )
    return text

# Diccionario de reglas iniciales (ajustadas al competence_framework.json)
RULES = {
    "emotional.regulation": {
        "keywords": ["descontrolo", "exploto", "pierdo el control"],
        "score": "weak"
    },
    "emotional.awareness": {
        "keywords": ["no se que siento", "confuso con mis emociones"],
        "score": "weak"
    },
    "emotional.frustration_tolerance": {
        "keywords": ["me frustro", "no tolero fallos", "pierdo paciencia"],
        "score": "weak"
    },
    "emotional.emotional_expression": {
        "keywords": ["no expreso mis emociones", "me lo guardo", "no muestro lo que siento"],
        "score": "weak"
    },
    "cognitive.flexibility": {
        "keywords": ["soy rigido", "me cuesta cambiar de idea"],
        "score": "weak"
    },
    "cognitive.rumination": {
        "keywords": ["darle vueltas", "rumiar", "pensando todo el tiempo"],
        "score": "high"
    },
    "cognitive.self_concept": {
        "keywords": ["soy inutil", "no valgo nada", "fracaso"],
        "score": "negative"
    },
    "cognitive.problem_solving": {
        "keywords": ["no encuentro soluciones", "me bloqueo con problemas"],
        "score": "weak"
    },
    "behavioral.activation": {
        "keywords": ["no inicio nada", "me cuesta empezar", "sin ganas de hacer cosas"],
        "score": "weak"
    },
    "behavioral.self_care": {
        "keywords": ["no duermo", "no me cuido", "no como bien"],
        "score": "weak"
    },
    "behavioral.avoidance_exposure": {
        "keywords": ["evito enfrentarme", "me escapo de los problemas"],
        "score": "avoidant"
    },
    "behavioral.goal_persistence": {
        "keywords": ["me rindo facilmente", "abandono metas"],
        "score": "weak"
    },
    "interpersonal.assertiveness": {
        "keywords": ["me callo", "no digo lo que pienso", "no se decir que no"],
        "score": "weak"
    },
    "interpersonal.social_support": {
        "keywords": ["no tengo apoyo", "nadie me ayuda"],
        "score": "low"
    },
    "interpersonal.conflict_resolution": {
        "keywords": ["evito conflictos", "no resuelvo discusiones"],
        "score": "weak"
    },
    "interpersonal.empathy": {
        "keywords": ["sin empatia", "me cuesta ponerme en su lugar"],
        "score": "weak"
    }
}

class MCPAnalyzer:
    def __init__(self):
        self.rules = RULES

    def analyze_chunk(self, chunk_text: str) -> list:
        results = []
        clean_text = normalize_text(chunk_text)

        for competence, rule in self.rules.items():
            for kw in rule["keywords"]:
                if kw in clean_text:
                    results.append({
                        "competence": competence,
                        "score": rule["score"],
                        "evidence": chunk_text
                    })
                    break  # evitar duplicados por competencia

        return results

if __name__ == "__main__":
    analyzer = MCPAnalyzer()

    # Prueba 1 - Emocional
    text_emocional = "Me descontrolo y exploto por cualquier cosa."
    print("Prueba Emocional:", analyzer.analyze_chunk(text_emocional))

    # Prueba 2 - Cognitiva
    text_cognitiva = "No puedo parar de darle vueltas a todo, me quedo rumiando horas."
    print("Prueba Cognitiva:", analyzer.analyze_chunk(text_cognitiva))

    # Prueba 3 - Conductual
    text_conductual = "Procrastino mucho, no inicio nada y me cuesta mantener hábitos."
    print("Prueba Conductual:", analyzer.analyze_chunk(text_conductual))

    # Prueba 4 - Interpersonal
    text_interpersonal = "Me callo y nunca digo lo que pienso, me cuesta decir que no."
    print("Prueba Interpersonal:", analyzer.analyze_chunk(text_interpersonal))