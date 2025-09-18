from psy_supabase.mcp.mcp_report_integration import MCPReport
from psy_supabase.mcp.mcp_longitudinal import MCPLongitudinal
from psy_supabase.db.mcp_storage import save_mcp_session


def generar_informe(
    resumen: str,
    emociones: str,
    recomendaciones: str,
    chunks: list[str],
    session_id: str,
    user_id: str = "demo_user_001",
) -> str:
    """
    Genera un informe clínico completo en formato Markdown mejorado,
    incluyendo perfil MCP puntual y evolución MCP longitudinal.
    También intenta guardar el perfil MCP en Supabase para persistencia.

    Args:
        resumen (str): Resumen del paciente.
        emociones (str): Señales emocionales.
        recomendaciones (str): Recomendaciones preliminares.
        chunks (list[str]): Fragmentos de texto de la sesión.
        session_id (str): Identificador de la sesión.
        user_id (str, opcional): Identificador del usuario/paciente (default: "demo_user_001").

    Returns:
        str: Informe en formato Markdown.
    """
    # 1. Perfil MCP puntual
    mcp = MCPReport()
    perfil_dict = mcp.generate_mcp_profile(chunks)
    perfil_md = mcp.to_markdown(perfil_dict)

    # 2. Guardar sesión y obtener evolución MCP (persistencia + cálculo)
    tracker = MCPLongitudinal()
    tracker.add_session(session_id, perfil_dict)

    # 2.1 Persistencia en Supabase (no bloqueante si falla)
    try:
        save_mcp_session(session_id, perfil_dict, user_id=user_id)
    except Exception as e:
        print(f"[WARN] No se pudo guardar en Supabase: {e}")

    evolucion = tracker.summarize()

    # --- Construcción evolución con indicadores ---
    if not evolucion:
        evolucion_md = "## Evolución MCP\n_No hay datos suficientes para mostrar tendencias._"
    else:
        lines = ["## Evolución MCP"]
        for competence, scores in evolucion.items():
            # Trend detection
            if len(scores) >= 2:
                if scores[-1] == scores[0]:
                    trend = "⚪ (estable)"
                elif scores[-1] > scores[0]:
                    trend = "⚠️ (empeora)"
                else:
                    trend = "✅ (mejora)"
            else:
                trend = ""
            evolution_str = " → ".join(scores)
            lines.append(f"- {competence}: {evolution_str} {trend}")
        evolucion_md = "\n".join(lines)

    # 3. Plantilla de informe clínico en Markdown
    return f"""
# 🧾 Informe Clínico de Sesión

---

## 📝 Resumen del paciente
{resumen}

## 💭 Señales emocionales
{emociones}

{perfil_md}

{evolucion_md}

## 🧩 Recomendaciones preliminares
{recomendaciones}

---
👨‍⚕️ *Este informe es un apoyo complementario. No sustituye la valoración profesional clínica.*
"""


# 👇 AÑADIDO: bloque de ejecución directa
if __name__ == "__main__":
    import os, datetime

    # Datos de prueba
    resumen = "Paciente refiere preocupación constante y dificultad para iniciar tareas."
    emociones = "Ansiedad elevada, frustración moderada."
    recomendaciones = "Introducir técnicas de respiración y planificación de micro‑tareas."
    chunks = [
        "Me cuesta mucho dejar de pensar en problemas una y otra vez.",
        "Siento ansiedad antes de iniciar tareas importantes.",
        "A veces no logro organizar mis actividades y me frustro."
    ]

    session_id = "session_test_001"

    # Generar informe
    informe = generar_informe(resumen, emociones, recomendaciones, chunks, session_id)

    # Imprimir en consola
    print(informe)

    # Guardar en reports/
    reports_dir = os.path.join(os.path.dirname(__file__), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_path = os.path.join(reports_dir, f"informe_session_test_{timestamp}.md")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(informe)

    print(f"[INFO] Informe guardado en {file_path}")