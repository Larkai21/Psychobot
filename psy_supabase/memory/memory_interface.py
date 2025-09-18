"""
Memory Interface for Layered Memory Architecture

Defines the common interface that all memory layers must implement for consistent
therapeutic memory management across short-term, medium-term, and long-term storage.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np


@dataclass
class MemoryItem:
    """Represents a memory item across all memory layers."""
    id: Optional[int] = None
    patient_id: str = ""
    session_id: Optional[str] = None
    content: str = ""
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = None
    created_at: Optional[datetime] = None
    memory_type: str = "general"  # short_term, medium_term, long_term
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RetrievalResult:
    """Represents a memory retrieval result with similarity scoring."""
    memory_item: MemoryItem
    similarity_score: float
    retrieval_context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.retrieval_context is None:
            self.retrieval_context = {}


class MemoryInterface(ABC):
    """
    Abstract base class defining the interface for all memory layers.
    
    Each memory layer (short-term, medium-term, long-term) must implement
    these methods to ensure consistent therapeutic memory management.
    """
    
    @abstractmethod
    def store(self, memory_item: MemoryItem) -> bool:
        """
        Store a memory item in this layer.
        
        Args:
            memory_item: The memory item to store
            
        Returns:
            bool: True if storage was successful, False otherwise
        """
        pass
    
    @abstractmethod
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
        Retrieve similar memories from this layer.
        
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
        pass
    
    @abstractmethod
    def delete(self, memory_id: int, patient_id: str) -> bool:
        """
        Delete a specific memory item.
        
        Args:
            memory_id: ID of the memory to delete
            patient_id: Patient identifier for authorization
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_memory_stats(self, patient_id: str) -> Dict[str, Any]:
        """
        Get statistics about memories for a patient in this layer.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            Dict[str, Any]: Statistics including count, date ranges, themes, etc.
        """
        pass
    
    @abstractmethod
    def consolidate(self, patient_id: str, **kwargs) -> int:
        """
        Perform layer-specific consolidation operations.
        
        For short-term: Clean up old session data
        For medium-term: Cluster weekly chunks into summaries  
        For long-term: Identify recurring themes across weeks
        
        Args:
            patient_id: Patient identifier
            **kwargs: Layer-specific consolidation parameters
            
        Returns:
            int: Number of items processed during consolidation
        """
        pass
    
    def get_layer_name(self) -> str:
        """Get the name of this memory layer."""
        return self.__class__.__name__.lower().replace('memory', '').replace('_', '')
    
    def validate_memory_item(self, memory_item: MemoryItem) -> bool:
        """
        Validate that a memory item meets layer requirements.
        
        Args:
            memory_item: Memory item to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        if not memory_item.patient_id:
            return False
        if not memory_item.content:
            return False
        if not memory_item.embedding or len(memory_item.embedding) == 0:
            return False
        return True
    
    def calculate_similarity(
        self, 
        embedding1: List[float], 
        embedding2: List[float]
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            float: Cosine similarity score (0-1)
        """
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            similarity = dot_product / (norm1 * norm2)
            
            # Ensure result is between 0 and 1
            return max(0.0, min(1.0, similarity))
            
        except Exception:
            return 0.0
    
    def filter_by_metadata(
        self, 
        memories: List[MemoryItem], 
        filters: Dict[str, Any]
    ) -> List[MemoryItem]:
        """
        Filter memories by metadata criteria.
        
        Args:
            memories: List of memory items to filter
            filters: Dictionary of metadata filters
            
        Returns:
            List[MemoryItem]: Filtered memory items
        """
        if not filters:
            return memories
            
        filtered = []
        for memory in memories:
            match = True
            for key, value in filters.items():
                if key not in memory.metadata:
                    match = False
                    break
                if memory.metadata[key] != value:
                    match = False
                    break
            if match:
                filtered.append(memory)
                
        return filtered


class MemoryLayerManager:
    """
    Manages coordination between different memory layers.
    
    Provides high-level operations that span multiple memory layers
    and handles layer-specific routing based on query context.
    """
    
    def __init__(self):
        self.short_term: Optional[MemoryInterface] = None
        self.medium_term: Optional[MemoryInterface] = None
        self.long_term: Optional[MemoryInterface] = None
    
    def register_layer(self, layer: MemoryInterface, layer_type: str):
        """Register a memory layer with the manager."""
        if layer_type == "short_term":
            self.short_term = layer
        elif layer_type == "medium_term":
            self.medium_term = layer
        elif layer_type == "long_term":
            self.long_term = layer
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")
    
    def route_query(
        self, 
        query_context: Dict[str, Any]
    ) -> List[MemoryInterface]:
        """
        Determine which memory layers to query based on context.
        
        Args:
            query_context: Context information about the query
            
        Returns:
            List[MemoryInterface]: Ordered list of layers to query
        """
        layers = []
        
        # Direct session queries -> short-term first
        if query_context.get("session_focused", False):
            if self.short_term:
                layers.append(self.short_term)
        
        # Pattern detection -> medium-term
        if query_context.get("pattern_detection", False):
            if self.medium_term:
                layers.append(self.medium_term)
        
        # Longitudinal/recurring themes -> long-term
        if query_context.get("longitudinal", False):
            if self.long_term:
                layers.append(self.long_term)
        
        # Default: query all available layers
        if not layers:
            for layer in [self.short_term, self.medium_term, self.long_term]:
                if layer:
                    layers.append(layer)
        
        return layers
    
    def multi_layer_retrieve(
        self,
        query_embedding: List[float],
        patient_id: str,
        query_context: Dict[str, Any],
        total_limit: int = 10
    ) -> List[RetrievalResult]:
        """
        Retrieve memories from multiple layers based on query context.
        
        Args:
            query_embedding: Vector embedding of the query
            patient_id: Patient identifier
            query_context: Context for layer routing
            total_limit: Maximum total results across all layers
            
        Returns:
            List[RetrievalResult]: Merged and ranked results from all layers
        """
        all_results = []
        layers = self.route_query(query_context)
        
        # Distribute limit across layers
        per_layer_limit = max(1, total_limit // len(layers)) if layers else 0
        
        for layer in layers:
            try:
                results = layer.retrieve(
                    query_embedding=query_embedding,
                    patient_id=patient_id,
                    limit=per_layer_limit
                )
                
                # Add layer context to results
                for result in results:
                    result.retrieval_context["layer"] = layer.get_layer_name()
                
                all_results.extend(results)
                
            except Exception as e:
                # Log error but continue with other layers
                print(f"Error retrieving from {layer.get_layer_name()}: {e}")
                continue
        
        # Sort by similarity score and limit
        all_results.sort(key=lambda x: x.similarity_score, reverse=True)
        return all_results[:total_limit]
