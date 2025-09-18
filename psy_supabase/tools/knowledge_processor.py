#!/usr/bin/env python3
"""
Knowledge Base Processor for Psychobot - Subfase 7.2
Procesa archivos .md y los convierte en chunks terapéuticos con metadatos.
"""

import os
import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any

class KnowledgeProcessor:
    def __init__(self, max_chunk_size: int = 300):
        self.max_chunk_size = max_chunk_size
        self.chunks = []
        
        # Mapeo de carpetas a temas principales
        self.folder_themes = {
            'life_changes': 'life_transitions',
            'psychoeducation': 'mental_health',
            'therapies': 'therapeutic_approaches',
            'selfhelp': 'self_care',
            'risk': 'crisis_support',
            'scales': 'assessment_tools'
        }
        
        # Mapeo de archivos específicos a emociones primarias
        self.emotion_mapping = {
            'anxiety': 'anxious',
            'depression': 'depressed',
            'ptsd': 'traumatized',
            'ocd': 'obsessive',
            'social_anxiety': 'anxious',
            'ageing': 'reflective',
            'postpartum_psychosis': 'distressed',
            'pregnancy': 'anticipatory',
            'university': 'stressed',
            'cbt': 'neutral',
            'act': 'accepting',
            'dbt': 'regulating',
            'mindfulness': 'calm',
            'sleep': 'restful',
            'exercise': 'energetic',
            'journaling': 'reflective',
            'relaxation': 'calm',
            'when_to_seek_help': 'concerned',
            'crisis_signposting': 'urgent',
            'phq9': 'neutral',
            'gad7': 'neutral'
        }

    def clean_text(self, text: str) -> str:
        """Limpia el texto eliminando markdown y caracteres especiales."""
        # Eliminar headers markdown
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        # Eliminar enlaces markdown
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        # Eliminar énfasis markdown
        text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^\*]+)\*', r'\1', text)
        # Eliminar listas markdown
        text = re.sub(r'^[\-\*\+]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)
        # Limpiar espacios extra
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def split_into_chunks(self, text: str) -> List[str]:
        """Divide el texto en chunks de 2-4 frases máximo."""
        # Dividir por párrafos primero
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks = []
        
        for paragraph in paragraphs:
            # Dividir por frases usando puntos, signos de exclamación y interrogación
            sentences = re.split(r'[.!?]+\s+', paragraph)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            current_chunk = ""
            sentence_count = 0
            
            for sentence in sentences:
                # Si añadir esta frase excede el límite de tokens o frases
                potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
                
                if (len(potential_chunk) > self.max_chunk_size or sentence_count >= 4) and current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                    sentence_count = 1
                else:
                    current_chunk = potential_chunk
                    sentence_count += 1
            
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if len(chunk) > 20]  # Filtrar chunks muy cortos

    def get_theme_from_path(self, file_path: Path) -> str:
        """Extrae el tema basado en la carpeta y nombre del archivo."""
        folder_name = file_path.parent.name
        file_name = file_path.stem
        
        # Usar el nombre del archivo como tema específico
        return file_name

    def get_primary_emotion(self, file_path: Path) -> str:
        """Determina la emoción primaria basada en el archivo."""
        file_name = file_path.stem
        return self.emotion_mapping.get(file_name, 'neutral')

    def process_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Procesa un archivo markdown individual."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Error leyendo {file_path}: {e}")
            return []

        if not content.strip():
            return []

        # Limpiar y dividir en chunks
        clean_content = self.clean_text(content)
        chunks = self.split_into_chunks(clean_content)
        
        # Crear metadatos
        theme = self.get_theme_from_path(file_path)
        primary_emotion = self.get_primary_emotion(file_path)
        folder_name = file_path.parent.name
        
        processed_chunks = []
        for i, chunk_text in enumerate(chunks):
            chunk_data = {
                "text": chunk_text,
                "source_file": str(file_path.relative_to(Path("knowledge_base"))),
                "theme": theme,
                "primary_emotion": primary_emotion,
                "tags": [],
                "chunk_id": f"{theme}_{i+1}"
            }
            
            # Añadir metadatos especiales según la carpeta
            if folder_name == "risk":
                chunk_data["critical"] = True
            elif folder_name == "scales":
                chunk_data["scale"] = True
            
            processed_chunks.append(chunk_data)
        
        return processed_chunks

    def process_knowledge_base(self, kb_path: str = "knowledge_base") -> Dict[str, Any]:
        """Procesa toda la knowledge base."""
        kb_path = Path(kb_path)
        
        if not kb_path.exists():
            raise FileNotFoundError(f"La carpeta {kb_path} no existe")
        
        all_chunks = []
        stats = {
            "total_files": 0,
            "total_chunks": 0,
            "files_by_category": {},
            "chunks_by_category": {}
        }
        
        # Recorrer todas las subcarpetas
        for folder in kb_path.iterdir():
            if not folder.is_dir():
                continue
                
            folder_name = folder.name
            stats["files_by_category"][folder_name] = 0
            stats["chunks_by_category"][folder_name] = 0
            
            # Procesar archivos .md en cada carpeta
            for md_file in folder.glob("*.md"):
                print(f"Procesando: {md_file}")
                chunks = self.process_file(md_file)
                all_chunks.extend(chunks)
                
                stats["total_files"] += 1
                stats["files_by_category"][folder_name] += 1
                stats["chunks_by_category"][folder_name] += len(chunks)
        
        stats["total_chunks"] = len(all_chunks)
        
        return {
            "metadata": {
                "version": "1.0",
                "processed_at": "2024-01-01T00:00:00Z",  # Puedes usar datetime.now().isoformat()
                "total_chunks": stats["total_chunks"],
                "max_chunk_size": self.max_chunk_size
            },
            "statistics": stats,
            "chunks": all_chunks
        }

    def save_processed_chunks(self, output_path: str = "knowledge_base/processed_chunks.json"):
        """Guarda los chunks procesados en un archivo JSON."""
        result = self.process_knowledge_base()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Procesamiento completado!")
        print(f"📁 Archivo guardado en: {output_path}")
        print(f"📊 Estadísticas:")
        print(f"   - Total archivos procesados: {result['statistics']['total_files']}")
        print(f"   - Total chunks generados: {result['statistics']['total_chunks']}")
        print(f"   - Chunks por categoría:")
        for category, count in result['statistics']['chunks_by_category'].items():
            print(f"     • {category}: {count} chunks")

def main():
    parser = argparse.ArgumentParser(description="Procesa la knowledge base de Psychobot")
    parser.add_argument("--max-chunk-size", type=int, default=300, 
                       help="Tamaño máximo de cada chunk en caracteres (default: 300)")
    parser.add_argument("--kb-path", default="knowledge_base", 
                       help="Ruta a la carpeta knowledge_base (default: knowledge_base)")
    parser.add_argument("--output", default="knowledge_base/processed_chunks.json",
                       help="Archivo de salida (default: knowledge_base/processed_chunks.json)")
    
    args = parser.parse_args()
    
    processor = KnowledgeProcessor(max_chunk_size=args.max_chunk_size)
    
    try:
        processor.save_processed_chunks(args.output)
    except Exception as e:
        print(f"❌ Error durante el procesamiento: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())