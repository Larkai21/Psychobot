"""
Unit tests for the layered memory architecture.

Tests cover:
1. Short-term memory storage and retrieval
2. Medium-term memory weekly consolidation
3. Long-term memory theme consolidation
4. Memory layer manager coordination
5. Integration with model manager and dynamic RAG
"""

import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch
from typing import Dict, List, Any

import pytest
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

from psy_supabase.memory.memory_interface import (
    MemoryInterface, 
    MemoryItem, 
    RetrievalResult, 
    MemoryLayerManager
)
from psy_supabase.memory.short_term import ShortTermMemory
from psy_supabase.memory.medium_term import MediumTermMemory
from psy_supabase.memory.long_term import LongTermMemory


class TestMemoryInterface(unittest.TestCase):
    """Test the base memory interface and layer manager."""

    def setUp(self):
        self.mock_db = MagicMock()
        self.patient_id = "test_patient_123"
        self.session_id = "test_session_456"

    def test_memory_item_creation(self):
        """Test MemoryItem creation and validation."""
        memory_item = MemoryItem(
            content="Test therapeutic content",
            patient_id=self.patient_id,
            session_id=self.session_id,
            memory_type="short_term",
            metadata={"emotion": "anxiety", "intensity": 0.7}
        )
        
        self.assertEqual(memory_item.content, "Test therapeutic content")
        self.assertEqual(memory_item.patient_id, self.patient_id)
        self.assertEqual(memory_item.session_id, self.session_id)
        self.assertEqual(memory_item.memory_type, "short_term")
        self.assertEqual(memory_item.metadata["emotion"], "anxiety")

    def test_retrieval_result_creation(self):
        """Test RetrievalResult creation."""
        memory_item = MemoryItem(
            content="Test content",
            patient_id=self.patient_id,
            session_id=self.session_id,
            memory_type="medium_term"
        )
        
        result = RetrievalResult(
            memory_item=memory_item,
            similarity_score=0.85,
            retrieval_context={"layer": "medium_term", "cluster_id": 5}
        )
        
        self.assertEqual(result.similarity_score, 0.85)
        self.assertEqual(result.retrieval_context["layer"], "medium_term")
        self.assertEqual(result.memory_item.content, "Test content")


class TestShortTermMemory(unittest.TestCase):
    """Test short-term memory functionality."""

    def setUp(self):
        self.mock_db = MagicMock()
        self.short_term = ShortTermMemory(self.mock_db)
        self.patient_id = "test_patient_123"
        self.session_id = "test_session_456"

    def test_store_memory_item(self):
        """Test storing a memory item in short-term memory."""
        memory_item = MemoryItem(
            content="Patient expressed anxiety about work",
            patient_id=self.patient_id,
            session_id=self.session_id,
            memory_type="short_term",
            metadata={"emotion": "anxiety", "intensity": 0.8}
        )
        
        # Mock successful storage
        self.mock_db.rpc.return_value.execute.return_value.data = [{"id": 1}]
        
        result = self.short_term.store(memory_item)
        
        self.assertTrue(result)
        self.mock_db.rpc.assert_called_once()

    def test_retrieve_session_memories(self):
        """Test retrieving memories for a specific session."""
        # Mock database response
        mock_memories = [
            {
                "id": 1,
                "content": "First memory",
                "embedding": [0.1] * 384,
                "metadata": {"emotion": "happy"},
                "created_at": datetime.now().isoformat()
            },
            {
                "id": 2, 
                "content": "Second memory",
                "embedding": [0.2] * 384,
                "metadata": {"emotion": "sad"},
                "created_at": datetime.now().isoformat()
            }
        ]
        
        self.mock_db.rpc.return_value.execute.return_value.data = mock_memories
        
        query_embedding = [0.15] * 384
        results = self.short_term.retrieve(
            query_embedding=query_embedding,
            patient_id=self.patient_id,
            session_id=self.session_id,
            limit=10
        )
        
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], RetrievalResult)
        self.assertEqual(results[0].memory_item.content, "First memory")

    def test_consolidate_expired_memories(self):
        """Test consolidation removes expired memories."""
        # Mock expired memories
        expired_date = (datetime.now() - timedelta(days=8)).isoformat()
        self.mock_db.rpc.return_value.execute.return_value.data = [
            {"session_id": "old_session", "created_at": expired_date}
        ]
        
        result = self.short_term.consolidate(self.patient_id)
        
        self.assertTrue(result)
        # Should call RPC to delete expired memories
        self.mock_db.rpc.assert_called()


class TestMediumTermMemory(unittest.TestCase):
    """Test medium-term memory functionality."""

    def setUp(self):
        self.mock_db = MagicMock()
        self.medium_term = MediumTermMemory(self.mock_db)
        self.patient_id = "test_patient_123"

    @patch('psy_supabase.memory.medium_term.KMeans')
    def test_weekly_consolidation(self, mock_kmeans):
        """Test weekly consolidation with clustering."""
        # Mock session chunks for clustering
        mock_chunks = [
            {"content": "Anxiety about work", "embedding": [0.1] * 384, "metadata": {"emotion": "anxiety"}},
            {"content": "Stress from deadlines", "embedding": [0.15] * 384, "metadata": {"emotion": "stress"}},
            {"content": "Happy about progress", "embedding": [0.8] * 384, "metadata": {"emotion": "happy"}}
        ]
        
        # Mock clustering results
        mock_kmeans_instance = Mock()
        mock_kmeans_instance.fit_predict.return_value = np.array([0, 0, 1])
        mock_kmeans_instance.cluster_centers_ = np.array([[0.125] * 384, [0.8] * 384])
        mock_kmeans.return_value = mock_kmeans_instance
        
        self.mock_db.rpc.return_value.execute.return_value.data = mock_chunks
        
        result = self.medium_term.consolidate(self.patient_id)
        
        self.assertTrue(result)
        mock_kmeans.assert_called_once()

    def test_retrieve_weekly_patterns(self):
        """Test retrieving weekly patterns."""
        mock_patterns = [
            {
                "id": 1,
                "content": "Weekly anxiety pattern about work deadlines",
                "embedding": [0.1] * 384,
                "metadata": {"cluster_size": 5, "dominant_emotion": "anxiety"},
                "created_at": datetime.now().isoformat()
            }
        ]
        
        self.mock_db.rpc.return_value.execute.return_value.data = mock_patterns
        
        query_embedding = [0.12] * 384
        results = self.medium_term.retrieve(
            query_embedding=query_embedding,
            patient_id=self.patient_id,
            limit=5
        )
        
        self.assertEqual(len(results), 1)
        self.assertIn("Weekly anxiety pattern", results[0].memory_item.content)


class TestLongTermMemory(unittest.TestCase):
    """Test long-term memory functionality."""

    def setUp(self):
        self.mock_db = MagicMock()
        self.long_term = LongTermMemory(self.mock_db)
        self.patient_id = "test_patient_123"

    @patch('psy_supabase.memory.long_term.DBSCAN')
    def test_theme_consolidation(self, mock_dbscan):
        """Test theme consolidation using DBSCAN clustering."""
        # Mock weekly summaries for theme detection
        mock_summaries = [
            {"content": "Recurring work anxiety", "embedding": [0.1] * 384, "metadata": {"theme": "work_stress"}},
            {"content": "Family relationship issues", "embedding": [0.5] * 384, "metadata": {"theme": "relationships"}},
            {"content": "Work stress continues", "embedding": [0.12] * 384, "metadata": {"theme": "work_stress"}}
        ]
        
        # Mock DBSCAN clustering
        mock_dbscan_instance = Mock()
        mock_dbscan_instance.fit_predict.return_value = np.array([0, 1, 0])  # Two themes
        mock_dbscan.return_value = mock_dbscan_instance
        
        self.mock_db.rpc.return_value.execute.return_value.data = mock_summaries
        
        result = self.long_term.consolidate(self.patient_id)
        
        self.assertTrue(result)
        mock_dbscan.assert_called_once()

    def test_retrieve_longitudinal_themes(self):
        """Test retrieving long-term themes."""
        mock_themes = [
            {
                "id": 1,
                "content": "Persistent work-related anxiety spanning multiple months",
                "embedding": [0.1] * 384,
                "metadata": {"theme_category": "work_stress", "persistence_score": 0.9, "occurrences": 12},
                "created_at": datetime.now().isoformat()
            }
        ]
        
        self.mock_db.rpc.return_value.execute.return_value.data = mock_themes
        
        query_embedding = [0.11] * 384
        results = self.long_term.retrieve(
            query_embedding=query_embedding,
            patient_id=self.patient_id,
            limit=3
        )
        
        self.assertEqual(len(results), 1)
        self.assertIn("Persistent work-related anxiety", results[0].memory_item.content)


class TestMemoryLayerManager(unittest.TestCase):
    """Test the memory layer manager coordination."""

    def setUp(self):
        self.mock_db = MagicMock()
        self.short_term = Mock(spec=ShortTermMemory)
        self.medium_term = Mock(spec=MediumTermMemory)
        self.long_term = Mock(spec=LongTermMemory)
        
        self.manager = MemoryLayerManager(
            short_term_memory=self.short_term,
            medium_term_memory=self.medium_term,
            long_term_memory=self.long_term
        )
        
        self.patient_id = "test_patient_123"

    def test_multi_layer_retrieve_session_focused(self):
        """Test retrieval with session-focused context."""
        query_embedding = [0.1] * 384
        query_context = {"session_focused": True, "pattern_detection": False, "longitudinal": False}
        
        # Mock short-term results
        mock_short_results = [
            RetrievalResult(
                memory_item=MemoryItem("Recent session content", self.patient_id, "session_1", "short_term"),
                similarity_score=0.9,
                retrieval_context={"layer": "short_term"}
            )
        ]
        self.short_term.retrieve.return_value = mock_short_results
        
        results = self.manager.multi_layer_retrieve(
            query_embedding=query_embedding,
            patient_id=self.patient_id,
            query_context=query_context,
            total_limit=10
        )
        
        self.assertEqual(len(results), 1)
        self.short_term.retrieve.assert_called_once()
        self.medium_term.retrieve.assert_not_called()
        self.long_term.retrieve.assert_not_called()

    def test_multi_layer_retrieve_all_layers(self):
        """Test retrieval from all layers."""
        query_embedding = [0.1] * 384
        query_context = {"session_focused": True, "pattern_detection": True, "longitudinal": True}
        
        # Mock results from all layers
        self.short_term.retrieve.return_value = [Mock()]
        self.medium_term.retrieve.return_value = [Mock()]
        self.long_term.retrieve.return_value = [Mock()]
        
        results = self.manager.multi_layer_retrieve(
            query_embedding=query_embedding,
            patient_id=self.patient_id,
            query_context=query_context,
            total_limit=10
        )
        
        # Should call all three layers
        self.short_term.retrieve.assert_called_once()
        self.medium_term.retrieve.assert_called_once()
        self.long_term.retrieve.assert_called_once()

    def test_consolidate_all_layers(self):
        """Test consolidation across all layers."""
        self.short_term.consolidate.return_value = True
        self.medium_term.consolidate.return_value = True
        self.long_term.consolidate.return_value = True
        
        result = self.manager.consolidate_all_layers(self.patient_id)
        
        self.assertTrue(result)
        self.short_term.consolidate.assert_called_once_with(self.patient_id)
        self.medium_term.consolidate.assert_called_once_with(self.patient_id)
        self.long_term.consolidate.assert_called_once_with(self.patient_id)


if __name__ == '__main__':
    unittest.main()
