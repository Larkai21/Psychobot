# core/report_generator.py

from psy_supabase.mcp.mcp_report_integration import MCPReport
from psy_supabase.mcp.mcp_longitudinal import MCPLongitudinal
from psy_supabase.db.mcp_storage import save_mcp_session


def generar_informe(resumen: str, emociones: str, recomendaciones: str, chunks: list[str], session_id: str, user_id: str = "demo_user_001") -> str:
    """
    Genera un informe clínico completo incluyendo perfil MCP puntual y evolución MCP longitudinal.
    También guarda el perfil MCP en Supabase para persistencia.
    
    Args:
        resumen (str): Resumen de la sesión
        emociones (str): Señales emocionales detectadas
        recomendaciones (str): Recomendaciones clínicas
        chunks (list[str]): Fragmentos de texto para análisis MCP
        session_id (str): Identificador único de la sesión
        user_id (str): ID del usuario/paciente (default: "demo_user_001")
    
    Returns:
        str: Informe completo en formato Markdown
    """
    # 1. Perfil MCP puntual
    mcp = MCPReport()
    perfil_dict = mcp.generate_mcp_profile(chunks)
    perfil_md = mcp.to_markdown(perfil_dict)

    # 2. Guardar sesión y obtener evolución MCP
    tracker = MCPLongitudinal()
    tracker.add_session(session_id, perfil_dict)
    
    # 2.1 Guardar perfil MCP en Supabase
    save_mcp_session(session_id, perfil_dict, user_id=user_id)

    evolucion_md = tracker.to_markdown_evolution()

    # 3. Armar el informe completo en markdown
    return f"""
# Informe de sesión

## Resumen
{resumen}

## Señales emocionales
{emociones}

{perfil_md}

{evolucion_md}

## Recomendaciones
{recomendaciones}
"""