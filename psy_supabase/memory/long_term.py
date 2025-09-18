"""
Long-Term Memory Layer

Stores consolidated themes across multiple weeks with recurrent psychological themes.
Provides longitudinal analysis and recurring pattern detection for therapeutic insights.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Set, TYPE_CHECKING
from datetime import datetime, timedelta, date
from dataclasses import asdict
from collections import Counter, defaultdict
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

from .memory_interface import MemoryInterface, MemoryItem, RetrievalResult
from ..utilities.utils import safe_execute

# Only import for type checking, not at runtime
if TYPE_CHECKING:
    from ..core.database import DatabaseManager

logger = logging.getLogger(__name__)

class LongTermMemory(MemoryInterface):
    """
    Long-term memory layer for therapeutic chatbot.
    
    Stores consolidated themes across multiple weeks with recurrent psychological patterns.
    Optimized for longitudinal analysis and recurring theme detection.
    """
    
    def __init__(
    self,
    database_manager: "DatabaseManager",  # ✅ string literal, Python lo acepta como hint
    theme_recurrence_threshold: int = 3,
    theme_similarity_threshold: float = 0.8
):
        """
        Initialize long-term memory layer.
        
        Args:
            database_manager: Database manager instance
            theme_recurrence_threshold: Minimum occurrences to consider a theme recurring
            theme_similarity_threshold: Similarity threshold for theme clustering
        """
        self.db = database_manager
        self.theme_recurrence_threshold = theme_recurrence_threshold
        self.theme_similarity_threshold = theme_similarity_threshold
        self.table_name = "long_term_memory"
        
        # Predefined therapeutic theme categories
        self.theme_categories = {
            "ansiedad": ["ansiedad", "preocupación", "miedo", "pánico", "nerviosismo"],
            "depresión": ["depresión", "tristeza", "desesperanza", "vacío", "melancolía"],
            "relaciones": ["familia", "pareja", "amigos", "relaciones", "soledad"],
            "trabajo": ["trabajo", "carrera", "estrés laboral", "jefe", "compañeros"],
            "autoestima": ["autoestima", "confianza", "valía personal", "autoimagen"],
            "trauma": ["trauma", "abuso", "violencia", "pérdida", "duelo"],
            "adicciones": ["alcohol", "drogas", "adicción", "dependencia"],
            "salud": ["salud", "enfermedad", "dolor", "síntomas", "médico"],
            "identidad": ["identidad", "propósito", "sentido", "quién soy"],
            "cambio": ["cambio", "transición", "adaptación", "crecimiento"]
        }
        
    def store(self, memory_item: MemoryItem) -> bool:
        """
        Store a memory item in long-term memory.
        
        Args:
            memory_item: The memory item to store
            
        Returns:
            bool: True if storage was successful, False otherwise
        """
        try:
            if not self.validate_memory_item(memory_item):
                logger.error("Invalid memory item for long-term storage")
                return False
            
            # Add memory type to metadata
            if memory_item.metadata is None:
                memory_item.metadata = {}
            memory_item.metadata["memory_type"] = "long_term"
            memory_item.metadata["stored_at"] = datetime.now().isoformat()
            
            # Extract theme information from metadata
            theme_category = memory_item.metadata.get("theme_category", "general")
            recurrence_count = memory_item.metadata.get("recurrence_count", 1)
            first_occurrence = memory_item.metadata.get("first_occurrence")
            last_occurrence = memory_item.metadata.get("last_occurrence", datetime.now().date())
            consolidation_source = memory_item.metadata.get("consolidation_source", "theme_clustering")
            
            # Convert dates to proper format
            if isinstance(first_occurrence, str):
                first_occurrence = datetime.fromisoformat(first_occurrence).date()
            elif first_occurrence is None:
                first_occurrence = datetime.now().date()
                
            if isinstance(last_occurrence, str):
                last_occurrence = datetime.fromisoformat(last_occurrence).date()
            elif isinstance(last_occurrence, datetime):
                last_occurrence = last_occurrence.date()
            
            # Store using database RPC function
            result = safe_execute(
                lambda: self.db.supabase.rpc(
                    "store_long_term_memory",
                    {
                        "p_schema_name": self.db.schema_name,
                        "p_patient_id": memory_item.patient_id,
                        "p_session_id": memory_item.session_id,
                        "p_content": memory_item.content,
                        "p_embedding": memory_item.embedding,
                        "p_metadata": memory_item.metadata,
                        "p_theme_category": theme_category,
                        "p_recurrence_count": recurrence_count,
                        "p_first_occurrence": first_occurrence.isoformat(),
                        "p_last_occurrence": last_occurrence.isoformat(),
                        "p_consolidation_source": consolidation_source
                    }
                ).execute()
            )
            
            if result and result.data:
                memory_id = result.data
                logger.info(f"Stored long-term memory item {memory_id} for patient {memory_item.patient_id}")
                return True
            else:
                logger.error("Failed to store long-term memory item")
                return False
                
        except Exception as e:
            logger.error(f"Error storing long-term memory: {e}")
            return False
    
    def retrieve(
        self, 
        query_embedding: List[float], 
        patient_id: str,
        session_id: Optional[str] = None,
        limit: int = 5,
        min_similarity: float = 0.1,
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve similar memories from long-term memory.
        
        Args:
            query_embedding: Vector embedding of the query
            patient_id: Patient identifier for RLS filtering
            session_id: Optional session filter
            limit: Maximum number of results to return
            min_similarity: Minimum similarity threshold
            metadata_filters: Optional metadata-based filters
            
        Returns:
            List[RetrievalResult]: Ranked list of similar memories
        """
        try:
            # Prepare metadata filters
            metadata_filter_json = None
            if metadata_filters:
                metadata_filter_json = metadata_filters
            
            # Retrieve using database RPC function
            result = safe_execute(
                lambda: self.db.supabase.rpc(
                    "retrieve_memories_by_similarity",
                    {
                        "p_schema_name": self.db.schema_name,
                        "p_memory_table": self.table_name,
                        "p_query_embedding": query_embedding,
                        "p_patient_id": patient_id,
                        "p_session_id": session_id,
                        "p_limit": limit,
                        "p_min_similarity": min_similarity,
                        "p_metadata_filters": metadata_filter_json
                    }
                ).execute()
            )
            
            if not result or not result.data:
                return []
            
            # Convert results to RetrievalResult objects
            retrieval_results = []
            for row in result.data:
                memory_item = MemoryItem(
                    id=row["id"],
                    patient_id=row["patient_id"],
                    session_id=row["session_id"],
                    content=row["content"],
                    embedding=row["embedding"],
                    metadata=row["metadata"] or {},
                    created_at=datetime.fromisoformat(row["created_at"].replace('Z', '+00:00')) if row["created_at"] else None,
                    memory_type="long_term"
                )
                
                retrieval_result = RetrievalResult(
                    memory_item=memory_item,
                    similarity_score=row["similarity"],
                    retrieval_context={
                        "layer": "long_term",
                        "longitudinal": True,
                        "retrieved_at": datetime.now().isoformat()
                    }
                )
                
                retrieval_results.append(retrieval_result)
            
            logger.info(f"Retrieved {len(retrieval_results)} long-term memories for patient {patient_id}")
            return retrieval_results
            
        except Exception as e:
            logger.error(f"Error retrieving long-term memories: {e}")
            return []
    
    def delete(self, memory_id: int, patient_id: str) -> bool:
        """
        Delete a specific long-term memory item.
        
        Args:
            memory_id: ID of the memory to delete
            patient_id: Patient identifier for authorization
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        try:
            result = safe_execute(
                lambda: self.db.supabase.table(f"{self.db.schema_name}.{self.table_name}")
                .delete()
                .eq("id", memory_id)
                .eq("patient_id", patient_id)
                .execute()
            )
            
            if result and result.data:
                logger.info(f"Deleted long-term memory {memory_id} for patient {patient_id}")
                return True
            else:
                logger.warning(f"No long-term memory found with ID {memory_id} for patient {patient_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting long-term memory: {e}")
            return False
    
    def get_memory_stats(self, patient_id: str) -> Dict[str, Any]:
        """
        Get statistics about long-term memories for a patient.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            Dict[str, Any]: Statistics including themes, recurrence patterns, etc.
        """
        try:
            result = safe_execute(
                lambda: self.db.supabase.rpc(
                    "get_memory_layer_stats",
                    {
                        "p_schema_name": self.db.schema_name,
                        "p_memory_table": self.table_name,
                        "p_patient_id": patient_id
                    }
                ).execute()
            )
            
            if result and result.data and len(result.data) > 0:
                stats = result.data[0]
                
                # Get additional long-term specific stats
                theme_stats = self._get_theme_statistics(patient_id)
                
                return {
                    "layer": "long_term",
                    "total_memories": stats.get("total_memories", 0),
                    "oldest_memory": stats.get("oldest_memory"),
                    "newest_memory": stats.get("newest_memory"),
                    "avg_content_length": stats.get("avg_content_length", 0),
                    "unique_sessions": stats.get("unique_sessions", 0),
                    "emotion_distribution": stats.get("metadata_summary", {}),
                    "theme_statistics": theme_stats,
                    "recurrence_threshold": self.theme_recurrence_threshold,
                    "similarity_threshold": self.theme_similarity_threshold
                }
            else:
                return {
                    "layer": "long_term",
                    "total_memories": 0,
                    "oldest_memory": None,
                    "newest_memory": None,
                    "avg_content_length": 0,
                    "unique_sessions": 0,
                    "emotion_distribution": {},
                    "theme_statistics": {},
                    "recurrence_threshold": self.theme_recurrence_threshold,
                    "similarity_threshold": self.theme_similarity_threshold
                }
                
        except Exception as e:
            logger.error(f"Error getting long-term memory stats: {e}")
            return {"layer": "long_term", "error": str(e)}
    
    def consolidate(self, patient_id: str, **kwargs) -> int:
        """
        Perform theme consolidation from medium-term memories.
        
        Args:
            patient_id: Patient identifier
            **kwargs: Additional parameters (months_back, force_consolidation)
            
        Returns:
            int: Number of items processed during consolidation
        """
        try:
            months_back = kwargs.get("months_back", 6)  # Process last 6 months by default
            force_consolidation = kwargs.get("force_consolidation", False)
            
            processed_count = 0
            
            # Get medium-term memories for theme analysis
            medium_term_memories = self._get_medium_term_memories(patient_id, months_back)
            
            if len(medium_term_memories) < self.theme_recurrence_threshold:
                logger.info(f"Not enough medium-term memories ({len(medium_term_memories)}) for theme consolidation")
                return 0
            
            # Identify recurring themes
            recurring_themes = self._identify_recurring_themes(medium_term_memories)
            
            # Consolidate each recurring theme
            for theme_data in recurring_themes:
                if self._consolidate_theme(patient_id, theme_data):
                    processed_count += len(theme_data["memories"])
            
            return processed_count
            
        except Exception as e:
            logger.error(f"Error consolidating long-term memories: {e}")
            return 0
    
    def _get_medium_term_memories(self, patient_id: str, months_back: int) -> List[Dict[str, Any]]:
        """
        Get medium-term memories for theme analysis.
        
        Args:
            patient_id: Patient identifier
            months_back: Number of months to look back
            
        Returns:
            List[Dict[str, Any]]: List of medium-term memory records
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=months_back * 30)
            
            result = safe_execute(
                lambda: self.db.supabase.table(f"{self.db.schema_name}.medium_term_memory")
                .select("*")
                .eq("patient_id", patient_id)
                .gte("created_at", cutoff_date.isoformat())
                .execute()
            )
            
            return result.data if result and result.data else []
            
        except Exception as e:
            logger.error(f"Error getting medium-term memories: {e}")
            return []
    
    def _identify_recurring_themes(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Identify recurring themes from medium-term memories.
        
        Args:
            memories: List of medium-term memory records
            
        Returns:
            List[Dict[str, Any]]: List of recurring theme data
        """
        try:
            # Group memories by theme categories
            theme_groups = defaultdict(list)
            
            for memory in memories:
                metadata = memory.get("metadata", {})
                theme = self._categorize_theme(memory["content"], metadata)
                theme_groups[theme].append(memory)
            
            # Filter themes by recurrence threshold
            recurring_themes = []
            for theme, theme_memories in theme_groups.items():
                if len(theme_memories) >= self.theme_recurrence_threshold:
                    # Further cluster by semantic similarity
                    clusters = self._cluster_theme_memories(theme_memories)
                    
                    for cluster in clusters:
                        if len(cluster) >= self.theme_recurrence_threshold:
                            recurring_themes.append({
                                "theme_category": theme,
                                "memories": cluster,
                                "recurrence_count": len(cluster),
                                "first_occurrence": min([mem["created_at"] for mem in cluster]),
                                "last_occurrence": max([mem["created_at"] for mem in cluster])
                            })
            
            return recurring_themes
            
        except Exception as e:
            logger.error(f"Error identifying recurring themes: {e}")
            return []
    
    def _categorize_theme(self, content: str, metadata: Dict[str, Any]) -> str:
        """
        Categorize content into therapeutic themes.
        
        Args:
            content: Memory content text
            metadata: Memory metadata
            
        Returns:
            str: Theme category
        """
        content_lower = content.lower()
        
        # Check metadata first
        if metadata.get("dominant_theme"):
            theme = metadata["dominant_theme"].lower()
            for category, keywords in self.theme_categories.items():
                if any(keyword in theme for keyword in keywords):
                    return category
        
        # Check content for theme keywords
        theme_scores = {}
        for category, keywords in self.theme_categories.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            if score > 0:
                theme_scores[category] = score
        
        if theme_scores:
            return max(theme_scores, key=theme_scores.get)
        
        return "general"
    
    def _cluster_theme_memories(self, theme_memories: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Cluster memories within a theme by semantic similarity.
        
        Args:
            theme_memories: List of memories with the same theme
            
        Returns:
            List[List[Dict[str, Any]]]: List of memory clusters
        """
        try:
            if len(theme_memories) < 2:
                return [theme_memories]
            
            # Extract embeddings
            embeddings = []
            valid_memories = []
            
            for memory in theme_memories:
                if memory.get("embedding"):
                    embeddings.append(memory["embedding"])
                    valid_memories.append(memory)
            
            if len(embeddings) < 2:
                return [valid_memories] if valid_memories else []
            
            # Use DBSCAN for density-based clustering
            embedding_matrix = np.array(embeddings)
            
            # Calculate epsilon based on similarity threshold
            eps = 1 - self.theme_similarity_threshold
            
            clustering = DBSCAN(
                eps=eps, 
                min_samples=max(2, self.theme_recurrence_threshold),
                metric='cosine'
            )
            
            cluster_labels = clustering.fit_predict(embedding_matrix)
            
            # Group memories by cluster
            clusters = defaultdict(list)
            for i, label in enumerate(cluster_labels):
                if label != -1:  # -1 is noise in DBSCAN
                    clusters[label].append(valid_memories[i])
            
            # Include noise points as individual clusters if they meet threshold
            noise_points = [valid_memories[i] for i, label in enumerate(cluster_labels) if label == -1]
            if len(noise_points) >= self.theme_recurrence_threshold:
                clusters[len(clusters)] = noise_points
            
            return list(clusters.values())
            
        except Exception as e:
            logger.error(f"Error clustering theme memories: {e}")
            # Fallback: return all memories as one cluster
            return [theme_memories]
    
    def _consolidate_theme(self, patient_id: str, theme_data: Dict[str, Any]) -> bool:
        """
        Consolidate a recurring theme into long-term memory.
        
        Args:
            patient_id: Patient identifier
            theme_data: Theme data with memories and metadata
            
        Returns:
            bool: True if consolidation was successful
        """
        try:
            memories = theme_data["memories"]
            theme_category = theme_data["theme_category"]
            
            # Create consolidated content
            consolidated_content = self._create_theme_summary(memories, theme_category)
            
            # Calculate representative embedding
            representative_embedding = self._calculate_theme_embedding(memories)
            
            # Create consolidated metadata
            consolidated_metadata = self._create_theme_metadata(memories, theme_data)
            
            # Create memory item
            memory_item = MemoryItem(
                patient_id=patient_id,
                session_id=None,  # Consolidated across sessions
                content=consolidated_content,
                embedding=representative_embedding,
                metadata=consolidated_metadata,
                created_at=datetime.now(),
                memory_type="long_term"
            )
            
            # Store the consolidated theme
            return self.store(memory_item)
            
        except Exception as e:
            logger.error(f"Error consolidating theme: {e}")
            return False
    
    def _create_theme_summary(self, memories: List[Dict[str, Any]], theme_category: str) -> str:
        """Create a summary for a recurring theme."""
        try:
            # Extract key content patterns
            contents = [mem["content"] for mem in memories if mem.get("content")]
            
            # Create theme-specific summary
            summary = f"Tema recurrente: {theme_category.title()}. "
            summary += f"Identificado en {len(memories)} períodos diferentes. "
            
            # Add representative content snippets
            if contents:
                # Take first few contents as examples
                examples = contents[:3]
                summary += "Ejemplos: " + " | ".join(examples)
                
                if len(contents) > 3:
                    summary += f" ... y {len(contents) - 3} más."
            
            return summary
            
        except Exception as e:
            logger.error(f"Error creating theme summary: {e}")
            return f"Tema recurrente: {theme_category}"
    
    def _calculate_theme_embedding(self, memories: List[Dict[str, Any]]) -> List[float]:
        """Calculate representative embedding for a theme."""
        try:
            embeddings = [mem["embedding"] for mem in memories if mem.get("embedding")]
            
            if not embeddings:
                return []
            
            # Calculate weighted centroid (more recent memories have higher weight)
            embedding_matrix = np.array(embeddings)
            
            # Create weights based on recency
            dates = [datetime.fromisoformat(mem["created_at"].replace('Z', '+00:00')) for mem in memories]
            max_date = max(dates)
            weights = [(max_date - date).days + 1 for date in dates]
            weights = np.array(weights) / sum(weights)
            
            # Calculate weighted centroid
            weighted_centroid = np.average(embedding_matrix, axis=0, weights=weights)
            
            return weighted_centroid.tolist()
            
        except Exception as e:
            logger.error(f"Error calculating theme embedding: {e}")
            # Fallback to simple average
            embeddings = [mem["embedding"] for mem in memories if mem.get("embedding")]
            if embeddings:
                return np.mean(embeddings, axis=0).tolist()
            return []
    
    def _create_theme_metadata(self, memories: List[Dict[str, Any]], theme_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create consolidated metadata for a theme."""
        try:
            # Extract temporal information
            dates = [datetime.fromisoformat(mem["created_at"].replace('Z', '+00:00')).date() for mem in memories]
            first_occurrence = min(dates)
            last_occurrence = max(dates)
            
            # Calculate theme persistence
            date_range = (last_occurrence - first_occurrence).days
            persistence_score = len(memories) / max(1, date_range / 30)  # memories per month
            
            # Extract emotional patterns
            emotions = []
            polarities = []
            intensities = []
            
            for memory in memories:
                metadata = memory.get("metadata", {})
                if metadata.get("dominant_emotion"):
                    emotions.append(metadata["dominant_emotion"])
                if metadata.get("avg_polarity") is not None:
                    polarities.append(metadata["avg_polarity"])
                if metadata.get("avg_intensity") is not None:
                    intensities.append(metadata["avg_intensity"])
            
            # Create consolidated metadata
            consolidated_metadata = {
                "theme_category": theme_data["theme_category"],
                "recurrence_count": theme_data["recurrence_count"],
                "first_occurrence": first_occurrence.isoformat(),
                "last_occurrence": last_occurrence.isoformat(),
                "persistence_score": persistence_score,
                "consolidation_source": "theme_clustering",
                "source_memory_count": len(memories),
                "date_range_days": date_range
            }
            
            # Add emotional analysis
            if emotions:
                consolidated_metadata["dominant_emotions"] = dict(Counter(emotions))
                consolidated_metadata["primary_emotion"] = max(set(emotions), key=emotions.count)
            
            if polarities:
                consolidated_metadata["avg_polarity"] = np.mean(polarities)
                consolidated_metadata["polarity_trend"] = "improving" if polarities[-1] > polarities[0] else "declining"
            
            if intensities:
                consolidated_metadata["avg_intensity"] = np.mean(intensities)
                consolidated_metadata["intensity_trend"] = "increasing" if intensities[-1] > intensities[0] else "decreasing"
            
            return consolidated_metadata
            
        except Exception as e:
            logger.error(f"Error creating theme metadata: {e}")
            return {
                "theme_category": theme_data.get("theme_category", "general"),
                "recurrence_count": theme_data.get("recurrence_count", 1),
                "consolidation_source": "theme_clustering"
            }
    
    def _get_theme_statistics(self, patient_id: str) -> Dict[str, Any]:
        """Get detailed theme statistics for a patient."""
        try:
            result = safe_execute(
                lambda: self.db.supabase.table(f"{self.db.schema_name}.{self.table_name}")
                .select("theme_category, recurrence_count, first_occurrence, last_occurrence, metadata")
                .eq("patient_id", patient_id)
                .execute()
            )
            
            if not result or not result.data:
                return {}
            
            # Analyze theme patterns
            themes = result.data
            theme_stats = {
                "total_themes": len(themes),
                "theme_categories": {},
                "most_persistent_theme": None,
                "most_recent_theme": None,
                "average_recurrence": 0
            }
            
            # Calculate statistics
            recurrence_counts = []
            for theme in themes:
                category = theme["theme_category"]
                recurrence = theme["recurrence_count"]
                
                if category not in theme_stats["theme_categories"]:
                    theme_stats["theme_categories"][category] = {
                        "count": 0,
                        "total_recurrence": 0,
                        "latest_occurrence": None
                    }
                
                theme_stats["theme_categories"][category]["count"] += 1
                theme_stats["theme_categories"][category]["total_recurrence"] += recurrence
                
                # Track latest occurrence
                last_occ = theme["last_occurrence"]
                if (theme_stats["theme_categories"][category]["latest_occurrence"] is None or 
                    last_occ > theme_stats["theme_categories"][category]["latest_occurrence"]):
                    theme_stats["theme_categories"][category]["latest_occurrence"] = last_occ
                
                recurrence_counts.append(recurrence)
            
            # Calculate averages
            if recurrence_counts:
                theme_stats["average_recurrence"] = np.mean(recurrence_counts)
            
            # Find most persistent and recent themes
            if themes:
                theme_stats["most_persistent_theme"] = max(themes, key=lambda x: x["recurrence_count"])
                theme_stats["most_recent_theme"] = max(themes, key=lambda x: x["last_occurrence"])
            
            return theme_stats
            
        except Exception as e:
            logger.error(f"Error getting theme statistics: {e}")
            return {}
    
    def get_recurring_themes(self, patient_id: str, min_recurrence: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get all recurring themes for a patient.
        
        Args:
            patient_id: Patient identifier
            min_recurrence: Minimum recurrence count filter
            
        Returns:
            List[Dict[str, Any]]: List of recurring themes
        """
        try:
            min_rec = min_recurrence or self.theme_recurrence_threshold
            
            result = safe_execute(
                lambda: self.db.supabase.table(f"{self.db.schema_name}.{self.table_name}")
                .select("*")
                .eq("patient_id", patient_id)
                .gte("recurrence_count", min_rec)
                .order("recurrence_count", desc=True)
                .execute()
            )
            
            return result.data if result and result.data else []
            
        except Exception as e:
            logger.error(f"Error getting recurring themes: {e}")
            return []
    
    def get_theme_timeline(self, patient_id: str, theme_category: str) -> List[Dict[str, Any]]:
        """
        Get timeline of a specific theme for a patient.
        
        Args:
            patient_id: Patient identifier
            theme_category: Theme category to analyze
            
        Returns:
            List[Dict[str, Any]]: Timeline of theme occurrences
        """
        try:
            result = safe_execute(
                lambda: self.db.supabase.table(f"{self.db.schema_name}.{self.table_name}")
                .select("*")
                .eq("patient_id", patient_id)
                .eq("theme_category", theme_category)
                .order("first_occurrence")
                .execute()
            )
            
            return result.data if result and result.data else []
            
        except Exception as e:
            logger.error(f"Error getting theme timeline: {e}")
            return []
