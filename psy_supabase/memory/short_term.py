"""
Short-Term Memory Layer

Stores detailed chunks from the last active session only.
Provides fast access to recent therapeutic interactions with automatic expiration.
"""

import logging
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from datetime import datetime, timedelta
from dataclasses import asdict

from .memory_interface import MemoryInterface, MemoryItem, RetrievalResult
from ..utilities.utils import safe_execute

# Only import for type checking, not at runtime
if TYPE_CHECKING:
    from ..core.database import DatabaseManager

logger = logging.getLogger(__name__)


class ShortTermMemory(MemoryInterface):
    """
    Short-term memory layer for therapeutic chatbot.
    
    Stores detailed chunks from the current and recent sessions with automatic expiration.
    Optimized for fast retrieval of recent therapeutic context.
    """
    
    def __init__(self, database_manager: "DatabaseManager", expiration_days: int = 7):
        """
        Initialize short-term memory layer.
        
        Args:
            database_manager: Database manager instance
            expiration_days: Number of days before memories expire
        """
        self.db = database_manager
        self.expiration_days = expiration_days
        self.table_name = "short_term_memory"
        
    def store(self, memory_item: MemoryItem) -> bool:
        """
        Store a memory item in short-term memory.
        
        Args:
            memory_item: The memory item to store
            
        Returns:
            bool: True if storage was successful, False otherwise
        """
        try:
            if not self.validate_memory_item(memory_item):
                logger.error("Invalid memory item for short-term storage")
                return False
            
            # Ensure session_id is provided for short-term memory
            if not memory_item.session_id:
                logger.error("Session ID required for short-term memory storage")
                return False
            
            # Add memory type to metadata
            if memory_item.metadata is None:
                memory_item.metadata = {}
            memory_item.metadata["memory_type"] = "short_term"
            memory_item.metadata["stored_at"] = datetime.now().isoformat()
            
            # Store using database RPC function
            result = safe_execute(
                lambda: self.db.supabase.rpc(
                    "store_short_term_memory",
                    {
                        "p_schema_name": self.db.schema_name,
                        "p_patient_id": memory_item.patient_id,
                        "p_session_id": memory_item.session_id,
                        "p_content": memory_item.content,
                        "p_embedding": memory_item.embedding,
                        "p_metadata": memory_item.metadata
                    }
                ).execute()
            )
            
            if result and result.data:
                memory_id = result.data
                logger.info(f"Stored short-term memory item {memory_id} for patient {memory_item.patient_id}")
                return True
            else:
                logger.error("Failed to store short-term memory item")
                return False
                
        except Exception as e:
            logger.error(f"Error storing short-term memory: {e}")
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
        Retrieve similar memories from short-term memory.
        
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
                    memory_type="short_term"
                )
                
                retrieval_result = RetrievalResult(
                    memory_item=memory_item,
                    similarity_score=row["similarity"],
                    retrieval_context={
                        "layer": "short_term",
                        "session_focused": True,
                        "retrieved_at": datetime.now().isoformat()
                    }
                )
                
                retrieval_results.append(retrieval_result)
            
            logger.info(f"Retrieved {len(retrieval_results)} short-term memories for patient {patient_id}")
            return retrieval_results
            
        except Exception as e:
            logger.error(f"Error retrieving short-term memories: {e}")
            return []
    
    def delete(self, memory_id: int, patient_id: str) -> bool:
        """
        Delete a specific short-term memory item.
        
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
                logger.info(f"Deleted short-term memory {memory_id} for patient {patient_id}")
                return True
            else:
                logger.warning(f"No short-term memory found with ID {memory_id} for patient {patient_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting short-term memory: {e}")
            return False
    
    def get_memory_stats(self, patient_id: str) -> Dict[str, Any]:
        """
        Get statistics about short-term memories for a patient.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            Dict[str, Any]: Statistics including count, date ranges, sessions, etc.
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
                return {
                    "layer": "short_term",
                    "total_memories": stats.get("total_memories", 0),
                    "oldest_memory": stats.get("oldest_memory"),
                    "newest_memory": stats.get("newest_memory"),
                    "avg_content_length": stats.get("avg_content_length", 0),
                    "unique_sessions": stats.get("unique_sessions", 0),
                    "emotion_distribution": stats.get("metadata_summary", {}),
                    "expiration_days": self.expiration_days
                }
            else:
                return {
                    "layer": "short_term",
                    "total_memories": 0,
                    "oldest_memory": None,
                    "newest_memory": None,
                    "avg_content_length": 0,
                    "unique_sessions": 0,
                    "emotion_distribution": {},
                    "expiration_days": self.expiration_days
                }
                
        except Exception as e:
            logger.error(f"Error getting short-term memory stats: {e}")
            return {"layer": "short_term", "error": str(e)}
    
    def consolidate(self, patient_id: str, **kwargs) -> int:
        """
        Clean up expired short-term memories and prepare for consolidation.
        
        Args:
            patient_id: Patient identifier
            **kwargs: Additional parameters (session_id, force_cleanup)
            
        Returns:
            int: Number of items processed during consolidation
        """
        try:
            processed_count = 0
            
            # Clean up expired memories
            cleanup_result = safe_execute(
                lambda: self.db.supabase.rpc(
                    "cleanup_expired_short_term_memories",
                    {"p_schema_name": self.db.schema_name}
                ).execute()
            )
            
            if cleanup_result and cleanup_result.data:
                expired_count = cleanup_result.data
                processed_count += expired_count
                logger.info(f"Cleaned up {expired_count} expired short-term memories")
            
            # Optional: Archive old session data before expiration
            archive_threshold = kwargs.get("archive_threshold_days", 3)
            if archive_threshold < self.expiration_days:
                archive_count = self._archive_old_sessions(patient_id, archive_threshold)
                processed_count += archive_count
            
            return processed_count
            
        except Exception as e:
            logger.error(f"Error consolidating short-term memories: {e}")
            return 0
    
    def _archive_old_sessions(self, patient_id: str, threshold_days: int) -> int:
        """
        Archive old session data to medium-term memory.
        
        Args:
            patient_id: Patient identifier
            threshold_days: Days threshold for archiving
            
        Returns:
            int: Number of sessions archived
        """
        try:
            # Get sessions older than threshold
            cutoff_date = datetime.now() - timedelta(days=threshold_days)
            
            result = safe_execute(
                lambda: self.db.supabase.table(f"{self.db.schema_name}.{self.table_name}")
                .select("session_id, content, embedding, metadata, created_at")
                .eq("patient_id", patient_id)
                .lt("created_at", cutoff_date.isoformat())
                .execute()
            )
            
            if not result or not result.data:
                return 0
            
            # Group by session for archiving
            sessions_to_archive = {}
            for row in result.data:
                session_id = row["session_id"]
                if session_id not in sessions_to_archive:
                    sessions_to_archive[session_id] = []
                sessions_to_archive[session_id].append(row)
            
            archived_count = 0
            for session_id, session_memories in sessions_to_archive.items():
                if self._create_session_summary(patient_id, session_id, session_memories):
                    archived_count += 1
            
            logger.info(f"Archived {archived_count} sessions to medium-term memory")
            return archived_count
            
        except Exception as e:
            logger.error(f"Error archiving old sessions: {e}")
            return 0
    
    def _create_session_summary(
        self, 
        patient_id: str, 
        session_id: str, 
        memories: List[Dict[str, Any]]
    ) -> bool:
        """
        Create a session summary for medium-term storage.
        
        Args:
            patient_id: Patient identifier
            session_id: Session identifier
            memories: List of memory items from the session
            
        Returns:
            bool: True if summary was created successfully
        """
        try:
            # Combine content from all memories
            combined_content = " ".join([mem["content"] for mem in memories])
            
            # Average embeddings (simple approach)
            embeddings = [mem["embedding"] for mem in memories if mem["embedding"]]
            if not embeddings:
                return False
            
            avg_embedding = [
                sum(emb[i] for emb in embeddings) / len(embeddings)
                for i in range(len(embeddings[0]))
            ]
            
            # Combine metadata
            combined_metadata = {
                "session_summary": True,
                "source_memory_count": len(memories),
                "session_id": session_id,
                "archived_from": "short_term",
                "archived_at": datetime.now().isoformat()
            }
            
            # Extract dominant emotions and themes
            emotions = [mem["metadata"].get("primary_emotion") for mem in memories if mem.get("metadata")]
            emotions = [e for e in emotions if e]
            if emotions:
                combined_metadata["dominant_emotion"] = max(set(emotions), key=emotions.count)
            
            # Store in medium-term memory (would need medium-term memory instance)
            # For now, just log the summary creation
            logger.info(f"Created session summary for {session_id} with {len(memories)} memories")
            return True
            
        except Exception as e:
            logger.error(f"Error creating session summary: {e}")
            return False
    
    def get_active_sessions(self, patient_id: str, days_back: int = 1) -> List[str]:
        """
        Get list of active sessions for a patient.
        
        Args:
            patient_id: Patient identifier
            days_back: Number of days to look back
            
        Returns:
            List[str]: List of active session IDs
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            result = safe_execute(
                lambda: self.db.supabase.table(f"{self.db.schema_name}.{self.table_name}")
                .select("session_id")
                .eq("patient_id", patient_id)
                .gte("created_at", cutoff_date.isoformat())
                .execute()
            )
            
            if result and result.data:
                sessions = list(set([row["session_id"] for row in result.data]))
                return sessions
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting active sessions: {e}")
            return []
    
    def get_session_memories(
        self, 
        patient_id: str, 
        session_id: str
    ) -> List[MemoryItem]:
        """
        Get all memories for a specific session.
        
        Args:
            patient_id: Patient identifier
            session_id: Session identifier
            
        Returns:
            List[MemoryItem]: All memories from the session
        """
        try:
            result = safe_execute(
                lambda: self.db.supabase.table(f"{self.db.schema_name}.{self.table_name}")
                .select("*")
                .eq("patient_id", patient_id)
                .eq("session_id", session_id)
                .order("created_at")
                .execute()
            )
            
            if not result or not result.data:
                return []
            
            memories = []
            for row in result.data:
                memory_item = MemoryItem(
                    id=row["id"],
                    patient_id=row["patient_id"],
                    session_id=row["session_id"],
                    content=row["content"],
                    embedding=row["embedding"],
                    metadata=row["metadata"] or {},
                    created_at=datetime.fromisoformat(row["created_at"].replace('Z', '+00:00')) if row["created_at"] else None,
                    memory_type="short_term"
                )
                memories.append(memory_item)
            
            return memories
            
        except Exception as e:
            logger.error(f"Error getting session memories: {e}")
            return []