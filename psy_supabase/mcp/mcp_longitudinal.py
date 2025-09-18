import json
import os
from datetime import datetime
from pathlib import Path

# Ruta donde vamos a guardar el histórico localmente (luego se puede llevar a Supabase)
TRACKING_FILE = Path(__file__).parent / "mcp_history.json"


class MCPLongitudinal:
    def __init__(self):
        # Si no existe el histórico lo creamos vacío
        if not TRACKING_FILE.exists():
            with open(TRACKING_FILE, "w", encoding="utf-8") as f:
                json.dump([], f)

    def _load_history(self):
        with open(TRACKING_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_history(self, history):
        with open(TRACKING_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    def add_session(self, session_id: str, mcp_profile: dict):
        """
        Añade un resultado MCP de una sesión al histórico.
        """
        history = self._load_history()

        entry = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "mcp_profile": mcp_profile
        }

        history.append(entry)
        self._save_history(history)
        return entry

    def get_history(self):
        """
        Devuelve todo el historial MCP.
        """
        return self._load_history()

    def summarize(self):
        """
        Devuelve la evolución de cada competencia a lo largo del tiempo.
        """
        history = self._load_history()
        summary = {}

        for entry in history:
            profile = entry["mcp_profile"]
            for competence, score in profile.items():
                if competence not in summary:
                    summary[competence] = []
                summary[competence].append(score)

        return summary

    def to_markdown_evolution(self) -> str:
        """
        Convierte la evolución MCP a un bloque Markdown.
        """
        summary = self.summarize()
        if not summary:
            return "## Evolución MCP\n_No hay datos suficientes para mostrar tendencias._"

        lines = ["## Evolución MCP"]
        for competence, scores in summary.items():
            # Representamos la evolución como "low → moderate → high"
            evolution_str = " → ".join(scores)
            lines.append(f"- {competence}: {evolution_str}")

        return "\n".join(lines)


# Ejemplo rápido de uso
if __name__ == "__main__":
    tracker = MCPLongitudinal()

    # Simulamos dos sesiones con perfiles MCP distintos
    tracker.add_session("session_001", {
        "cognitive.rumination": "high",
        "emotional.regulation": "weak"
    })

    tracker.add_session("session_002", {
        "cognitive.rumination": "moderate",
        "emotional.regulation": "moderate"
    })

    print("Histórico completo:\n", tracker.get_history())
    print("\nEvolución competencias:\n", tracker.summarize())