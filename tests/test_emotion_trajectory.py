"""
Unit tests for Emotional Trajectory Tracking module.

This module provides comprehensive testing for the EmotionTrajectory class,
including emotional data recording, trajectory analysis, abrupt change detection,
graph export functionality, and clinical safety features.
"""

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest
import numpy as np

from psy_supabase.analytics.emotion_trajectory import (
    EmotionTrajectory,
    EmotionType,
    AlertSeverity,
    EmotionalDataPoint,
    TrajectoryAlert
)
from psy_supabase.core.database import DatabaseManager


class TestEmotionTrajectory(unittest.TestCase):
    """Test cases for EmotionTrajectory class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_db_manager = Mock(spec=DatabaseManager)
        self.mock_db_manager.user_id = "test_user_123"
        self.mock_db_manager.validate_access.return_value = True
        self.mock_db_manager.audit_access_attempt.return_value = True
        
        self.mock_supabase = Mock()
        self.mock_db_manager.supabase = self.mock_supabase
        
        self.trajectory = EmotionTrajectory(self.mock_db_manager)
        
        self.sample_chunk_metadata = {
            "primary_emotion": "happy",
            "intensity": 0.7,
            "polarity": 0.5,
            "theme": "therapeutic_progress"
        }
        
        self.sample_patient_id = "patient_123"
        self.sample_session_id = "session_456"

    def test_emotion_trajectory_initialization(self):
        """Test EmotionTrajectory initialization."""
        self.assertIsInstance(self.trajectory, EmotionTrajectory)
        self.assertEqual(self.trajectory.db_manager, self.mock_db_manager)
        self.assertIsInstance(self.trajectory._emotion_cache, dict)

    def test_record_chunk_emotion_success(self):
        """Test successful emotion recording from chunk metadata."""
        self.mock_supabase.rpc.return_value.execute.return_value.data = "trajectory_id_123"
        
        result = self.trajectory.record_chunk_emotion(
            self.sample_patient_id,
            self.sample_session_id,
            self.sample_chunk_metadata,
            "chunk_789"
        )
        
        self.assertTrue(result)
        self.mock_supabase.rpc.assert_called_once()
        self.mock_db_manager.validate_access.assert_called_with(
            "test_user_123", self.sample_patient_id, "INSERT"
        )

    def test_record_chunk_emotion_access_denied(self):
        """Test emotion recording with access denied."""
        self.mock_db_manager.validate_access.return_value = False
        
        result = self.trajectory.record_chunk_emotion(
            self.sample_patient_id,
            self.sample_session_id,
            self.sample_chunk_metadata
        )
        
        self.assertFalse(result)
        self.mock_supabase.rpc.assert_not_called()

    def test_get_patient_trajectory_success(self):
        """Test successful patient trajectory retrieval."""
        mock_trajectory_data = [
            EmotionalDataPoint(
                timestamp=datetime.now() - timedelta(days=1),
                emotion=EmotionType.HAPPY,
                intensity=0.7,
                polarity=0.5,
                session_id="session_1",
                chunk_id="chunk_1",
                theme="progress"
            ),
            EmotionalDataPoint(
                timestamp=datetime.now(),
                emotion=EmotionType.SAD,
                intensity=0.6,
                polarity=-0.3,
                session_id="session_2",
                chunk_id="chunk_2",
                theme="setback"
            )
        ]
        
        with patch.object(self.trajectory, '_retrieve_trajectory_data', return_value=mock_trajectory_data):
            result = self.trajectory.get_patient_trajectory(self.sample_patient_id, "1_month")
        
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], EmotionalDataPoint)
        self.assertEqual(result[0].emotion, EmotionType.HAPPY)

    def test_detect_abrupt_changes_negative_spike(self):
        """Test detection of abrupt negative emotional changes."""
        trajectory_data = []
        base_time = datetime.now()
        
        # Normal positive emotions
        for i in range(3):
            trajectory_data.append(EmotionalDataPoint(
                timestamp=base_time - timedelta(days=i+3),
                emotion=EmotionType.HAPPY,
                intensity=0.7,
                polarity=0.6,
                session_id=f"session_{i}",
                theme="normal"
            ))
        
        # Sudden negative emotions
        for i in range(3):
            trajectory_data.append(EmotionalDataPoint(
                timestamp=base_time - timedelta(days=i),
                emotion=EmotionType.SAD,
                intensity=0.8,
                polarity=-0.7,
                session_id=f"session_{i+3}",
                theme="crisis"
            ))
        
        with patch.object(self.trajectory, 'get_patient_trajectory', return_value=trajectory_data):
            with patch.object(self.trajectory, '_log_clinical_alert'):
                alerts = self.trajectory.detect_abrupt_changes(self.sample_patient_id, threshold=0.3)
        
        self.assertGreater(len(alerts), 0)
        abrupt_alerts = [a for a in alerts if a.alert_type == "abrupt_negative_change"]
        self.assertGreater(len(abrupt_alerts), 0)

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('os.makedirs')
    def test_export_graph_success(self, mock_makedirs, mock_close, mock_savefig):
        """Test successful graph export."""
        trajectory_data = [
            EmotionalDataPoint(
                timestamp=datetime.now() - timedelta(days=i),
                emotion=EmotionType.HAPPY,
                intensity=0.7,
                polarity=0.5,
                session_id=f"session_{i}",
                theme="test"
            ) for i in range(5)
        ]
        
        with patch.object(self.trajectory, 'get_patient_trajectory', return_value=trajectory_data):
            with tempfile.TemporaryDirectory() as temp_dir:
                result = self.trajectory.export_graph(
                    self.sample_patient_id,
                    timeframe="1_week",
                    format="png",
                    output_dir=temp_dir
                )
        
        self.assertIsNotNone(result)
        self.assertTrue(result.endswith('.png'))
        mock_savefig.assert_called_once()

    def test_get_trajectory_statistics_success(self):
        """Test successful trajectory statistics calculation."""
        trajectory_data = [
            EmotionalDataPoint(
                timestamp=datetime.now() - timedelta(days=i),
                emotion=EmotionType.HAPPY,
                intensity=0.7,
                polarity=0.5,
                session_id=f"session_{i}",
                theme="progress"
            ) for i in range(5)
        ]
        
        with patch.object(self.trajectory, 'get_patient_trajectory', return_value=trajectory_data):
            stats = self.trajectory.get_trajectory_statistics(self.sample_patient_id, "1_week")
        
        self.assertIn("total_data_points", stats)
        self.assertIn("polarity_stats", stats)
        self.assertEqual(stats["total_data_points"], 5)
        self.assertEqual(stats["polarity_stats"]["mean"], 0.5)

    def test_emotion_type_enum(self):
        """Test EmotionType enum values."""
        self.assertEqual(EmotionType.HAPPY.value, "happy")
        self.assertEqual(EmotionType.SAD.value, "sad")
        self.assertEqual(EmotionType.ANGRY.value, "angry")

    def test_end_to_end_workflow(self):
        """Test complete emotion tracking workflow."""
        patient_id = "patient_123"
        session_id = "session_456"
        
        # Record emotions
        self.mock_supabase.rpc.return_value.execute.return_value.data = "trajectory_id"
        
        result = self.trajectory.record_chunk_emotion(
            patient_id, session_id, self.sample_chunk_metadata, "chunk_1"
        )
        self.assertTrue(result)
        
        # Verify trajectory retrieval works
        mock_data = [EmotionalDataPoint(
            timestamp=datetime.now(),
            emotion=EmotionType.HAPPY,
            intensity=0.7,
            polarity=0.5,
            session_id=session_id,
            theme="test"
        )]
        
        with patch.object(self.trajectory, '_retrieve_trajectory_data', return_value=mock_data):
            trajectory = self.trajectory.get_patient_trajectory(patient_id, "1_week")
        
        self.assertEqual(len(trajectory), 1)


if __name__ == '__main__':
    unittest.main()
