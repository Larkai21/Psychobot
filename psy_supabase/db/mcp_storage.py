# psy_supabase/db/mcp_storage.py

import os
import json
from supabase import create_client, Client
from dotenv import load_dotenv

# Cargar variables desde .env
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env or environment variables")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def save_mcp_session(session_id: str, mcp_profile: dict, user_id: str = None):
    """
    Guarda el perfil MCP procesado en la tabla `mcp_sessions`.

    Args:
        session_id (str): Identificador único de sesión.
        mcp_profile (dict): Perfil MCP en formato dict.
        user_id (str, optional): Usuario asociado (default None).
    """
    data = {
        "session_id": session_id,
        "user_id": user_id,
        "mcp_profile": mcp_profile
    }
    try:
        response = supabase.table("mcp_sessions").insert(data).execute()
        return response
    except Exception as e:
        print(f"[ERROR] No se pudo guardar la sesión en Supabase: {e}")
        return None


def get_user_sessions(user_id: str):
    """
    Recupera todas las sesiones MCP de un usuario.
    """
    try:
        response = supabase.table("mcp_sessions").select("*").eq("user_id", user_id).execute()
        return response.data if response.data else []
    except Exception as e:
        print(f"[ERROR] No se pudieron recuperar sesiones: {e}")
        return []