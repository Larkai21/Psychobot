"""
Medium-Term Memory Layer

Stores weekly summaries from clustered embeddings of session chunks.
Provides pattern detection and weekly therapeutic progress tracking.
"""

import logging
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from datetime import datetime, timedelta, date
from dataclasses import asdict
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from .memory_interface import MemoryInterface, MemoryItem, RetrievalResult
from ..utilities.utils import safe_execute

# Only import for type checking, not at runtime
if TYPE_CHECKING:
    from ..core.database import DatabaseManager

logger = logging.getLogger(__name__)


class MediumTermMemory(MemoryInterface):
    """
    Medium-term memory layer for therapeutic chatbot.
    
    Stores weekly summaries created by clustering session chunks.
    Optimized for pattern detection and weekly therapeutic progress analysis.
    """
    
    def __init__(self, database_manager:"DatabaseManager", clustering_threshold: float = 0.7):
        """
        Initialize medium-term memory layer.
        
        Args:
            database_manager: Database manager instance
            clustering_threshold: Similarity threshold for clustering memories
        """
        self.db = database_manager
        self.clustering_threshold = clustering_threshold
        self.table_name = "medium_term_memory"
        
    def store(self, memory_item: MemoryItem) -> bool:
        """
        Store a memory item in medium-term memory.
        
        Args:
            memory_item: The memory item to store
            
        Returns:
            bool: True if storage was successful, False otherwise
        """
        try:
            if not self.validate_memory_item(memory_item):
                logger.error("Invalid memory item for medium-term storage")
                return False
            
            # Add memory type to metadata
            if memory_item.metadata is None:
                memory_item.metadata = {}
            memory_item.metadata["memory_type"] = "medium_term"
            memory_item.metadata["stored_at"] = datetime.now().isoformat()
            
            # Calculate week start date
            week_start = self._get_week_start(memory_item.created_at or datetime.now())
            
            # Get consolidation parameters from metadata
            consolidation_source = memory_item.metadata.get("consolidation_source", "weekly_clustering")
            source_memory_count = memory_item.metadata.get("source_memory_count", 1)
            
            # Store using database RPC function
            result = safe_execute(
                lambda: self.db.supabase.rpc(
                    "store_medium_term_memory",
                    {
                        "p_schema_name": self.db.schema_name,
                        "p_patient_id": memory_item.patient_id,
                        "p_session_id": memory_item.session_id,
                        "p_content": memory_item.content,
                        "p_embedding": memory_item.embedding,
                        "p_metadata": memory_item.metadata,
                        "p_week_start": week_start.isoformat(),
                        "p_consolidation_source": consolidation_source,
                        "p_source_memory_count": source_memory_count
                    }
                ).execute()
            )
            
            if result and result.data:
                memory_id = result.data
                logger.info(f"Stored medium-term memory item {memory_id} for patient {memory_item.patient_id}")
                return True
            else:
                logger.error("Failed to store medium-term memory item")
                return False
                
        except Exception as e:
            logger.error(f"Error storing medium-term memory: {e}")
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
        Retrieve similar memories from medium-term memory.
        
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
                    memory_type="medium_term"
                )
                
                retrieval_result = RetrievalResult(
                    memory_item=memory_item,
                    similarity_score=row["similarity"],
                    retrieval_context={
                        "layer": "medium_term",
                        "pattern_detection": True,
                        "retrieved_at": datetime.now().isoformat()
                    }
                )
                
                retrieval_results.append(retrieval_result)
            
            logger.info(f"Retrieved {len(retrieval_results)} medium-term memories for patient {patient_id}")
            return retrieval_results
            
        except Exception as e:
            logger.error(f"Error retrieving medium-term memories: {e}")
            return []
    
    def delete(self, memory_id: int, patient_id: str) -> bool:
        """
        Delete a specific medium-term memory item.
        
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
                logger.info(f"Deleted medium-term memory {memory_id} for patient {patient_id}")
                return True
            else:
                logger.warning(f"No medium-term memory found with ID {memory_id} for patient {patient_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting medium-term memory: {e}")
            return False
    
    def get_memory_stats(self, patient_id: str) -> Dict[str, Any]:
        """
        Get statistics about medium-term memories for a patient.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            Dict[str, Any]: Statistics including count, date ranges, weeks covered, etc.
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
                
                # Get additional medium-term specific stats
                weeks_covered = self._get_weeks_covered(patient_id)
                
                return {
                    "layer": "medium_term",
                    "total_memories": stats.get("total_memories", 0),
                    "oldest_memory": stats.get("oldest_memory"),
                    "newest_memory": stats.get("newest_memory"),
                    "avg_content_length": stats.get("avg_content_length", 0),
                    "unique_sessions": stats.get("unique_sessions", 0),
                    "emotion_distribution": stats.get("metadata_summary", {}),
                    "weeks_covered": weeks_covered,
                    "clustering_threshold": self.clustering_threshold
                }
            else:
                return {
                    "layer": "medium_term",
                    "total_memories": 0,
                    "oldest_memory": None,
                    "newest_memory": None,
                    "avg_content_length": 0,
                    "unique_sessions": 0,
                    "emotion_distribution": {},
                    "weeks_covered": 0,
                    "clustering_threshold": self.clustering_threshold
                }
                
        except Exception as e:
            logger.error(f"Error getting medium-term memory stats: {e}")
            return {"layer": "medium_term", "error": str(e)}
    
    def consolidate(self, patient_id: str, **kwargs) -> int:
        """
        Perform weekly clustering consolidation of short-term memories.
        
        Args:
            patient_id: Patient identifier
            **kwargs: Additional parameters (weeks_back, force_consolidation)
            
        Returns:
            int: Number of items processed during consolidation
        """
        try:
            weeks_back = kwargs.get("weeks_back", 4)  # Process last 4 weeks by default
            force_consolidation = kwargs.get("force_consolidation", False)
            
            processed_count = 0
            
            # Get weeks to process
            weeks_to_process = self._get_weeks_to_consolidate(patient_id, weeks_back, force_consolidation)
            
            for week_start in weeks_to_process:
                week_processed = self._consolidate_week(patient_id, week_start)
                if week_processed > 0:
                    processed_count += week_processed
                    logger.info(f"Consolidated {week_processed} memories for week {week_start}")
            
            return processed_count
            
        except Exception as e:
            logger.error(f"Error consolidating medium-term memories: {e}")
            return 0
    
    def _get_weeks_to_consolidate(
        self, 
        patient_id: str, 
        weeks_back: int, 
        force_consolidation: bool
    ) -> List[date]:
        """
        Get list of weeks that need consolidation.
        
        Args:
            patient_id: Patient identifier
            weeks_back: Number of weeks to look back
            force_consolidation: Whether to force re-consolidation
            
        Returns:
            List[date]: List of week start dates to consolidate
        """
        weeks_to_process = []
        current_date = datetime.now().date()
        
        for i in range(weeks_back):
            week_start = self._get_week_start(current_date - timedelta(weeks=i))
            
            # Check if week already consolidated (unless forcing)
            if not force_consolidation and self._is_week_consolidated(patient_id, week_start):
                continue
                
            # Check if there are memories to consolidate for this week
            if self._has_memories_for_week(patient_id, week_start):
                weeks_to_process.append(week_start)
        
        return weeks_to_process
    
    def _consolidate_week(self, patient_id: str, week_start: date) -> int:
        """
        Consolidate memories for a specific week using clustering.
        
        Args:
            patient_id: Patient identifier
            week_start: Start date of the week to consolidate
            
        Returns:
            int: Number of memories processed
        """
        try:
            # Get short-term memories for this week
            week_memories = self._get_short_term_memories_for_week(patient_id, week_start)
            
            if len(week_memories) < 2:
                # Not enough memories to cluster, store as single summary if any
                if len(week_memories) == 1:
                    return self._store_single_week_summary(patient_id, week_start, week_memories[0])
                return 0
            
            # Cluster memories by similarity
            clusters = self._cluster_memories(week_memories)
            
            # Create summaries for each cluster
            summaries_created = 0
            for cluster_memories in clusters:
                if self._create_cluster_summary(patient_id, week_start, cluster_memories):
                    summaries_created += 1
            
            return len(week_memories)
            
        except Exception as e:
            logger.error(f"Error consolidating week {week_start}: {e}")
            return 0
    
    def _get_short_term_memories_for_week(
        self, 
        patient_id: str, 
        week_start: date
    ) -> List[Dict[str, Any]]:
        """
        Get short-term memories for a specific week.
        
        Args:
            patient_id: Patient identifier
            week_start: Start date of the week
            
        Returns:
            List[Dict[str, Any]]: List of memory records
        """
        try:
            week_end = week_start + timedelta(days=7)
            
            result = safe_execute(
                lambda: self.db.supabase.table(f"{self.db.schema_name}.short_term_memory")
                .select("*")
                .eq("patient_id", patient_id)
                .gte("created_at", week_start.isoformat())
                .lt("created_at", week_end.isoformat())
                .execute()
            )
            
            return result.data if result and result.data else []
            
        except Exception as e:
            logger.error(f"Error getting short-term memories for week: {e}")
            return []
    
    def _cluster_memories(self, memories: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Cluster memories by embedding similarity.
        
        Args:
            memories: List of memory records with embeddings
            
        Returns:
            List[List[Dict[str, Any]]]: List of memory clusters
        """
        try:
            # Extract embeddings
            embeddings = []
            valid_memories = []
            
            for memory in memories:
                if memory.get("embedding"):
                    embeddings.append(memory["embedding"])
                    valid_memories.append(memory)
            
            if len(embeddings) < 2:
                return [valid_memories] if valid_memories else []
            
            # Convert to numpy array
            embedding_matrix = np.array(embeddings)
            
            # Determine optimal number of clusters
            n_clusters = min(max(2, len(embeddings) // 3), 5)  # 2-5 clusters
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embedding_matrix)
            
            # Group memories by cluster
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(valid_memories[i])
            
            # Filter clusters by similarity threshold
            filtered_clusters = []
            for cluster_memories in clusters.values():
                if self._validate_cluster_coherence(cluster_memories):
                    filtered_clusters.append(cluster_memories)
                else:
                    # Split incoherent cluster
                    for memory in cluster_memories:
                        filtered_clusters.append([memory])
            
            return filtered_clusters
            
        except Exception as e:
            logger.error(f"Error clustering memories: {e}")
            # Fallback: return each memory as its own cluster
            return [[memory] for memory in memories]
    
    def _validate_cluster_coherence(self, cluster_memories: List[Dict[str, Any]]) -> bool:
        """
        Validate that memories in a cluster are coherent.
        
        Args:
            cluster_memories: List of memories in the cluster
            
        Returns:
            bool: True if cluster is coherent, False otherwise
        """
        if len(cluster_memories) < 2:
            return True
        
        try:
            embeddings = [mem["embedding"] for mem in cluster_memories if mem.get("embedding")]
            if len(embeddings) < 2:
                return True
            
            # Calculate pairwise similarities
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                    similarities.append(sim)
            
            # Check if average similarity meets threshold
            avg_similarity = np.mean(similarities)
            return avg_similarity >= self.clustering_threshold
            
        except Exception as e:
            logger.error(f"Error validating cluster coherence: {e}")
            return False
    
    def _create_cluster_summary(
        self, 
        patient_id: str, 
        week_start: date, 
        cluster_memories: List[Dict[str, Any]]
    ) -> bool:
        """
        Create a summary for a cluster of memories.
        
        Args:
            patient_id: Patient identifier
            week_start: Start date of the week
            cluster_memories: List of memories in the cluster
            
        Returns:
            bool: True if summary was created successfully
        """
        try:
            # Combine content from cluster memories
            combined_content = self._combine_cluster_content(cluster_memories)
            
            # Calculate representative embedding
            representative_embedding = self._calculate_representative_embedding(cluster_memories)
            
            # Create consolidated metadata
            consolidated_metadata = self._consolidate_cluster_metadata(cluster_memories)
            consolidated_metadata.update({
                "consolidation_source": "weekly_clustering",
                "source_memory_count": len(cluster_memories),
                "cluster_coherence": self._calculate_cluster_coherence(cluster_memories),
                "week_start": week_start.isoformat()
            })
            
            # Create memory item
            memory_item = MemoryItem(
                patient_id=patient_id,
                session_id=None,  # Consolidated across sessions
                content=combined_content,
                embedding=representative_embedding,
                metadata=consolidated_metadata,
                created_at=datetime.now(),
                memory_type="medium_term"
            )
            
            # Store the summary
            return self.store(memory_item)
            
        except Exception as e:
            logger.error(f"Error creating cluster summary: {e}")
            return False
    
    def _combine_cluster_content(self, cluster_memories: List[Dict[str, Any]]) -> str:
        """Combine content from cluster memories into a coherent summary."""
        contents = [mem["content"] for mem in cluster_memories if mem.get("content")]
        
        # Simple combination - could be enhanced with NLP summarization
        if len(contents) == 1:
            return contents[0]
        
        # Group similar content and create summary
        combined = f"Resumen semanal de {len(contents)} interacciones: "
        combined += " | ".join(contents[:3])  # Limit to first 3 for brevity
        
        if len(contents) > 3:
            combined += f" ... y {len(contents) - 3} interacciones más."
        
        return combined
    
    def _calculate_representative_embedding(self, cluster_memories: List[Dict[str, Any]]) -> List[float]:
        """Calculate representative embedding for a cluster."""
        embeddings = [mem["embedding"] for mem in cluster_memories if mem.get("embedding")]
        
        if not embeddings:
            return []
        
        # Calculate centroid
        embedding_matrix = np.array(embeddings)
        centroid = np.mean(embedding_matrix, axis=0)
        
        return centroid.tolist()
    
    def _consolidate_cluster_metadata(self, cluster_memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Consolidate metadata from cluster memories."""
        consolidated = {}
        
        # Collect emotions
        emotions = []
        themes = []
        polarities = []
        intensities = []
        
        for memory in cluster_memories:
            metadata = memory.get("metadata", {})
            if metadata.get("primary_emotion"):
                emotions.append(metadata["primary_emotion"])
            if metadata.get("theme"):
                themes.append(metadata["theme"])
            if metadata.get("polarity") is not None:
                polarities.append(metadata["polarity"])
            if metadata.get("intensity") is not None:
                intensities.append(metadata["intensity"])
        
        # Calculate consolidated values
        if emotions:
            consolidated["dominant_emotion"] = max(set(emotions), key=emotions.count)
            consolidated["emotion_distribution"] = {
                emotion: emotions.count(emotion) for emotion in set(emotions)
            }
        
        if themes:
            consolidated["dominant_theme"] = max(set(themes), key=themes.count)
        
        if polarities:
            consolidated["avg_polarity"] = np.mean(polarities)
        
        if intensities:
            consolidated["avg_intensity"] = np.mean(intensities)
        
        return consolidated
    
    def _calculate_cluster_coherence(self, cluster_memories: List[Dict[str, Any]]) -> float:
        """Calculate coherence score for a cluster."""
        embeddings = [mem["embedding"] for mem in cluster_memories if mem.get("embedding")]
        
        if len(embeddings) < 2:
            return 1.0
        
        try:
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                    similarities.append(sim)
            
            return float(np.mean(similarities))
            
        except Exception:
            return 0.0
    
    def _store_single_week_summary(
        self, 
        patient_id: str, 
        week_start: date, 
        memory: Dict[str, Any]
    ) -> int:
        """Store a single memory as a week summary."""
        try:
            metadata = memory.get("metadata", {}).copy()
            metadata.update({
                "consolidation_source": "single_memory_summary",
                "source_memory_count": 1,
                "week_start": week_start.isoformat()
            })
            
            memory_item = MemoryItem(
                patient_id=patient_id,
                session_id=memory.get("session_id"),
                content=memory["content"],
                embedding=memory["embedding"],
                metadata=metadata,
                created_at=datetime.now(),
                memory_type="medium_term"
            )
            
            return 1 if self.store(memory_item) else 0
            
        except Exception as e:
            logger.error(f"Error storing single week summary: {e}")
            return 0
    
    def _get_week_start(self, date_obj: datetime) -> date:
        """Get the start of the week (Monday) for a given date."""
        if isinstance(date_obj, datetime):
            date_obj = date_obj.date()
        
        days_since_monday = date_obj.weekday()
        week_start = date_obj - timedelta(days=days_since_monday)
        return week_start
    
    def _get_weeks_covered(self, patient_id: str) -> int:
        """Get number of weeks covered by medium-term memories."""
        try:
            result = safe_execute(
                lambda: self.db.supabase.table(f"{self.db.schema_name}.{self.table_name}")
                .select("week_start")
                .eq("patient_id", patient_id)
                .execute()
            )
            
            if result and result.data:
                unique_weeks = set([row["week_start"] for row in result.data])
                return len(unique_weeks)
            
            return 0
            
        except Exception as e:
            logger.error(f"Error getting weeks covered: {e}")
            return 0
    
    def _is_week_consolidated(self, patient_id: str, week_start: date) -> bool:
        """Check if a week has already been consolidated."""
        try:
            result = safe_execute(
                lambda: self.db.supabase.table(f"{self.db.schema_name}.{self.table_name}")
                .select("id")
                .eq("patient_id", patient_id)
                .eq("week_start", week_start.isoformat())
                .limit(1)
                .execute()
            )
            
            return bool(result and result.data)
            
        except Exception as e:
            logger.error(f"Error checking week consolidation: {e}")
            return False
    
    def _has_memories_for_week(self, patient_id: str, week_start: date) -> bool:
        """Check if there are short-term memories for a week."""
        try:
            week_end = week_start + timedelta(days=7)
            
            result = safe_execute(
                lambda: self.db.supabase.table(f"{self.db.schema_name}.short_term_memory")
                .select("id")
                .eq("patient_id", patient_id)
                .gte("created_at", week_start.isoformat())
                .lt("created_at", week_end.isoformat())
                .limit(1)
                .execute()
            )
            
            return bool(result and result.data)
            
        except Exception as e:
            logger.error(f"Error checking memories for week: {e}")
            return False
