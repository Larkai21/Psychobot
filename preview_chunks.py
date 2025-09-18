# preview_chunks.py
import json
import random
from pathlib import Path

# Ruta al archivo JSON ya procesado
json_path = Path("knowledge_base/processed_chunks.json")

if not json_path.exists():
    print("❌ No se encontró:", json_path)
    print("Ejecuta primero knowledge_processor.py para generarlo.")
    exit(1)

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"✅ Total chunks: {data['metadata']['total_chunks']}")
print("📊 Chunks por categoría:")
for k, v in data["statistics"]["chunks_by_category"].items():
    print(f"   • {k}: {v}")

print("\n🎲 Ejemplo de 5 chunks aleatorios:\n")
for chunk in random.sample(data["chunks"], min(5, len(data["chunks"]))):
    print("-", chunk["text"])
    print("  (source:", chunk["source_file"], "| theme:", chunk["theme"], ")\n")