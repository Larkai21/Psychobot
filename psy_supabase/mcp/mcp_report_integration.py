from collections import defaultdict
from psy_supabase.mcp.mcp_analyzer import MCPAnalyzer


class MCPReport:
    def __init__(self):
        self.analyzer = MCPAnalyzer()

    def generate_mcp_profile(self, chunks: list[str]) -> dict:
        """
        Procesa una lista de chunks y devuelve un perfil MCP agregado.
        Retorna un dict con competencias detectadas y la última puntuación vista.
        """
        profile = defaultdict(list)

        for chunk in chunks:
            results = self.analyzer.analyze_chunk(chunk)
            for r in results:
                profile[r["competence"]].append(r["score"])

        # Reducir a una única entrada por competencia
        summary = {}
        for competence, scores in profile.items():
            # De momento nos quedamos con el valor "más frecuente"
            score = max(set(scores), key=scores.count)
            summary[competence] = score

        return summary

    def to_markdown(self, mcp_profile: dict) -> str:
        """
        Convierte el perfil MCP en un bloque de texto Markdown para informes.
        """
        if not mcp_profile:
            return "## Perfil MCP\n_No se han detectado competencias relevantes en esta sesión._"

        lines = ["## Perfil MCP"]
        for competence, score in mcp_profile.items():
            d1, d2 = competence.split(".")
            pretty_name = competence.replace(".", " → ")
            lines.append(f"- {pretty_name}: {score}")
        return "\n".join(lines)


# Ejemplo de uso
if __name__ == "__main__":
    report = MCPReport()
    test_chunks = [
        "Me descontrolo y exploto por cualquier cosa.",
        "No puedo parar de darle vueltas a todo, me quedo rumiando horas.",
        "Procrastino mucho, no inicio nada y me cuesta mantener hábitos.",
        "Me callo y nunca digo lo que pienso, me cuesta decir que no."
    ]

    profile = report.generate_mcp_profile(test_chunks)
    print("=== Perfil MCP dict ===")
    print(profile)

    md = report.to_markdown(profile)
    print("\n=== Perfil MCP markdown ===")
    print(md)