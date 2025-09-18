import pytest
from psy_supabase.mcp.mcp_analyzer import MCPAnalyzer

analyzer = MCPAnalyzer()

# Casos de prueba: cada tuple = (texto, competencia_esperada)
TEST_CASES = [
    ("Llevo toda la semana dándole vueltas a lo mismo, no puedo dejar de pensarlo...", "cognitive.rumination"),
    ("Siempre que discuto exploto y pierdo el control.", "emotional.regulation"),
    ("Me callo siempre y no sé decir que no.", "interpersonal.assertiveness"),
    ("La verdad es que no sé lo que siento, estoy confuso con mis emociones.", "emotional.awareness"),
    ("Soy muy rígido y me cuesta aceptar opiniones distintas.", "cognitive.flexibility"),
    ("No entiendo a los demás, me cuesta ponerme en su lugar.", "interpersonal.empathy"),
    ("Me siento inútil y sin valor.", "selfesteem.confidence"),
    ("No tengo metas en la vida, me falta motivación.", "motivation.goal_setting"),
    ("Estoy estresado y siento que no doy abasto.", "stress.management"),
    ("Soy optimista y valoro las cosas buenas que tengo.", "emotional.positivity"),
    ("Me siento solo y no logro mantener amistades.", "interpersonal.relationships"),
    ("Me distraigo con cualquier cosa, no me concentro.", "cognitive.attention"),
    ("Me bloqueo al elegir, no sé decidir qué hacer.", "decision.making"),
    ("Soy muy procrastinador y me dejo llevar por los impulsos.", "self.regulation"),
    ("Me recupero rápido de los problemas, me siento fuerte en la adversidad.", "resilience"),
    ("Siento que me falta sentido en la vida, no encuentro un propósito.", "values.purpose")
]

@pytest.mark.parametrize("text, expected_competence", TEST_CASES)
def test_analyzer_detects_competence(text, expected_competence):
    results = analyzer.analyze_chunk(text)
    competences = [r["competence"] for r in results]
    assert expected_competence in competences, f"'{expected_competence}' no fue detectada en: {text}"