"""
Comprehensive tests for clinical report generation functionality.

This module tests the ClinicalReportGenerator class and related functionality
including report generation, PDF export, role-based access control, and data redaction.
"""

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import pytest

try:
    import weasyprint
except ImportError:
    pytest.skip("Skipping report tests: 'weasyprint' not installed", allow_module_level=True)
from psy_supabase.core.database import DatabaseManager
from tests.helpers.database_test_base import DatabaseTestBase


class TestClinicalReportGenerator(DatabaseTestBase):
    """Test cases for clinical report generation."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        
        # Mock database manager
        self.mock_db = Mock(spec=DatabaseManager)
        self.mock_db.validate_access = Mock(return_value=True)
        self.mock_db.audit_access_attempt = Mock()
        
        # Test data
        self.patient_id = "test_patient_123"
        self.psychologist_id = "psychologist_456"
        self.admin_id = "admin_789"
        
        # Initialize report generator
        self.report_generator = ClinicalReportGenerator(self.mock_db, self.psychologist_id)

    def test_init_with_valid_user(self) -> None:
        """Test initialization with valid user."""
        generator = ClinicalReportGenerator(self.mock_db, self.psychologist_id)
        self.assertEqual(generator.db_manager, self.mock_db)
        self.assertEqual(generator.user_id, self.psychologist_id)

    def test_init_with_patient_user_raises_permission_error(self) -> None:
        """Test that patients cannot create report generators."""
        with self.assertRaises(PermissionError) as context:
            ClinicalReportGenerator(self.mock_db, self.patient_id)
        
        self.assertIn("Patients are not allowed to generate reports", str(context.exception))

    @patch('psy_supabase.analytics.reports.EmotionTrajectory')
    def test_generate_patient_report_success(self, mock_emotion_trajectory: Mock) -> None:
        """Test successful patient report generation."""
        # Mock emotion trajectory
        mock_trajectory_instance = Mock()
        mock_trajectory_instance.get_trajectory_statistics.return_value = {
            "total_data_points": 50,
            "avg_polarity": 0.3,
            "polarity_trend": "stable",
            "dominant_emotions": ["anxiety", "hope"],
            "concerning_episodes": 2
        }
        mock_emotion_trajectory.return_value = mock_trajectory_instance
        
        # Mock database responses
        self.mock_db.get_themed_chunks_for_analysis.return_value = [
            {
                "chunk_text": "I feel anxious about work",
                "psychological_theme": "workplace_anxiety",
                "embedding": [0.1] * 1536,
                "primary_emotion": "anxiety",
                "polarity": -0.2
            },
            {
                "chunk_text": "I'm worried about my relationship",
                "psychological_theme": "relationship_issues", 
                "embedding": [0.2] * 1536,
                "primary_emotion": "worry",
                "polarity": -0.1
            }
        ]
        
        self.mock_db.get_active_clinical_alerts.return_value = [
            {
                "alert_id": "alert_1",
                "severity": "high",
                "alert_type": "emotional_distress",
                "message": "Sustained negative emotional pattern detected",
                "created_at": datetime.now()
            }
        ]
        
        self.mock_db.count_patient_data_points.return_value = 50
        self.mock_db.count_patient_sessions.return_value = 10
        
        # Generate report
        report = self.report_generator.generate_patient_report(self.patient_id, "month")
        
        # Assertions
        self.assertIsNotNone(report)
        self.assertEqual(report["patient_id"], self.patient_id)
        self.assertEqual(report["timeframe"], "month")
        self.assertIn("executive_summary", report)
        self.assertIn("recurrent_themes", report)
        self.assertIn("emotional_trajectory", report)
        self.assertIn("clinical_alerts", report)
        self.assertIn("clinical_assessment", report)
        self.assertIn("recommendations", report)
        
        # Verify database calls
        self.mock_db.get_themed_chunks_for_analysis.assert_called_once()
        self.mock_db.get_active_clinical_alerts.assert_called_once_with(self.patient_id)

    def test_generate_patient_report_no_data(self) -> None:
        """Test report generation when no data is available."""
        # Mock empty responses
        self.mock_db.get_themed_chunks_for_analysis.return_value = []
        self.mock_db.get_active_clinical_alerts.return_value = []
        self.mock_db.count_patient_data_points.return_value = 0
        self.mock_db.count_patient_sessions.return_value = 0
        
        report = self.report_generator.generate_patient_report(self.patient_id, "month")
        
        self.assertIsNone(report)

    @patch('sklearn.cluster.DBSCAN')
    def test_summarize_recurrent_themes(self, mock_dbscan: Mock) -> None:
        """Test recurrent themes summarization with clustering."""
        # Mock clustering results
        mock_dbscan_instance = Mock()
        mock_dbscan_instance.fit_predict.return_value = [0, 0, 1, 1, -1]  # 2 clusters + noise
        mock_dbscan.return_value = mock_dbscan_instance
        
        # Mock chunk data
        chunks = [
            {"chunk_text": "Work stress", "psychological_theme": "workplace_anxiety", "embedding": [0.1] * 1536},
            {"chunk_text": "Job pressure", "psychological_theme": "workplace_anxiety", "embedding": [0.2] * 1536},
            {"chunk_text": "Relationship issues", "psychological_theme": "relationship_issues", "embedding": [0.3] * 1536},
            {"chunk_text": "Partner conflict", "psychological_theme": "relationship_issues", "embedding": [0.4] * 1536},
            {"chunk_text": "Random thought", "psychological_theme": "general", "embedding": [0.5] * 1536}
        ]
        
        themes = self.report_generator.summarize_recurrent_themes(chunks)
        
        self.assertIsInstance(themes, list)
        self.assertLessEqual(len(themes), 3)  # Top 3 themes
        
        for theme in themes:
            self.assertIn("theme", theme)
            self.assertIn("frequency", theme)
            self.assertIn("representative_examples", theme)
            self.assertIn("clinical_significance", theme)

    @patch('psy_supabase.analytics.reports.EmotionTrajectory')
    def test_summarize_emotional_trajectory(self, mock_emotion_trajectory: Mock) -> None:
        """Test emotional trajectory summarization."""
        # Mock trajectory statistics
        mock_trajectory_instance = Mock()
        mock_trajectory_instance.get_trajectory_statistics.return_value = {
            "total_data_points": 100,
            "avg_polarity": 0.2,
            "polarity_std": 0.4,
            "polarity_trend": "improving",
            "dominant_emotions": ["anxiety", "hope", "sadness"],
            "concerning_episodes": 3,
            "stability_score": 0.7
        }
        mock_emotion_trajectory.return_value = mock_trajectory_instance
        
        summary = self.report_generator.summarize_emotional_trajectory(self.patient_id, "month")
        
        self.assertIsInstance(summary, dict)
        self.assertIn("overall_trend", summary)
        self.assertIn("stability_assessment", summary)
        self.assertIn("dominant_emotions", summary)
        self.assertIn("concerning_patterns", summary)
        self.assertIn("progress_indicators", summary)

    def test_list_active_alerts(self) -> None:
        """Test active alerts listing and formatting."""
        # Mock alerts data
        mock_alerts = [
            {
                "alert_id": "alert_1",
                "severity": "high",
                "alert_type": "emotional_distress",
                "message": "Sustained negative emotional pattern",
                "created_at": datetime.now() - timedelta(hours=2),
                "metadata": {"pattern_duration": "3_days"}
            },
            {
                "alert_id": "alert_2", 
                "severity": "medium",
                "alert_type": "behavioral_change",
                "message": "Significant change in communication pattern",
                "created_at": datetime.now() - timedelta(hours=1),
                "metadata": {"change_type": "decreased_engagement"}
            }
        ]
        
        self.mock_db.get_active_clinical_alerts.return_value = mock_alerts
        
        alerts = self.report_generator.list_active_alerts(self.patient_id)
        
        self.assertEqual(len(alerts), 2)
        
        for alert in alerts:
            self.assertIn("alert_id", alert)
            self.assertIn("severity", alert)
            self.assertIn("priority_level", alert)
            self.assertIn("description", alert)
            self.assertIn("time_since_creation", alert)
            self.assertIn("recommended_action", alert)

    @patch('weasyprint.HTML')
    @patch('tempfile.NamedTemporaryFile')
    def test_export_to_pdf_success(self, mock_temp_file: Mock, mock_html: Mock) -> None:
        """Test successful PDF export."""
        # Mock temporary file
        mock_temp_file.return_value.__enter__.return_value.name = "/tmp/test_report.pdf"
        
        # Mock HTML rendering
        mock_html_instance = Mock()
        mock_html.return_value = mock_html_instance
        
        # Mock report data
        with patch.object(self.report_generator, 'generate_patient_report') as mock_generate:
            mock_generate.return_value = {
                "patient_id": self.patient_id,
                "timeframe": "month",
                "executive_summary": "Test summary",
                "recurrent_themes": [],
                "emotional_trajectory": {},
                "clinical_alerts": [],
                "clinical_assessment": {},
                "recommendations": []
            }
            
            pdf_path = self.report_generator.export_to_pdf(self.patient_id, "month")
            
            self.assertIsNotNone(pdf_path)
            mock_html_instance.write_pdf.assert_called_once()

    def test_export_to_pdf_no_data(self) -> None:
        """Test PDF export when no report data is available."""
        with patch.object(self.report_generator, 'generate_patient_report') as mock_generate:
            mock_generate.return_value = None
            
            pdf_path = self.report_generator.export_to_pdf(self.patient_id, "month")
            
            self.assertIsNone(pdf_path)

    def test_assess_clinical_risk_high(self) -> None:
        """Test clinical risk assessment for high-risk scenarios."""
        # High-risk scenario: multiple alerts + negative trajectory
        alerts = [
            {"severity": "high", "alert_type": "emotional_distress"},
            {"severity": "medium", "alert_type": "behavioral_change"}
        ]
        
        trajectory = {
            "overall_trend": "declining",
            "stability_assessment": "unstable",
            "concerning_patterns": ["sustained_negativity", "emotional_volatility"]
        }
        
        risk = self.report_generator._assess_clinical_risk(alerts, trajectory)
        
        self.assertEqual(risk["risk_level"], "high")
        self.assertIn("Multiple high-severity alerts", risk["rationale"])

    def test_assess_clinical_risk_low(self) -> None:
        """Test clinical risk assessment for low-risk scenarios."""
        # Low-risk scenario: no alerts + stable trajectory
        alerts = []
        
        trajectory = {
            "overall_trend": "stable",
            "stability_assessment": "stable", 
            "concerning_patterns": []
        }
        
        risk = self.report_generator._assess_clinical_risk(alerts, trajectory)
        
        self.assertEqual(risk["risk_level"], "low")
        self.assertIn("No active clinical alerts", risk["rationale"])

    def test_generate_clinical_recommendations(self) -> None:
        """Test clinical recommendations generation."""
        # Mock data for recommendations
        themes = [
            {"theme": "workplace_anxiety", "frequency": 15},
            {"theme": "relationship_issues", "frequency": 8}
        ]
        
        trajectory = {
            "overall_trend": "declining",
            "dominant_emotions": ["anxiety", "sadness"]
        }
        
        risk_assessment = {"risk_level": "medium"}
        
        recommendations = self.report_generator._generate_clinical_recommendations(
            themes, trajectory, risk_assessment
        )
        
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        
        for rec in recommendations:
            self.assertIn("category", rec)
            self.assertIn("recommendation", rec)
            self.assertIn("priority", rec)
            self.assertIn("rationale", rec)

    def test_redact_sensitive_information(self) -> None:
        """Test sensitive information redaction."""
        # Test data with potentially sensitive information
        test_data = {
            "patient_name": "John Doe",
            "phone": "555-123-4567",
            "email": "john@example.com",
            "safe_field": "This is safe",
            "nested": {
                "address": "123 Main St",
                "safe_nested": "Also safe"
            }
        }
        
        redacted = self.report_generator._redact_sensitive_information(test_data)
        
        # Check that sensitive fields are redacted
        self.assertEqual(redacted["patient_name"], "[REDACTED]")
        self.assertEqual(redacted["phone"], "[REDACTED]")
        self.assertEqual(redacted["email"], "[REDACTED]")
        self.assertEqual(redacted["nested"]["address"], "[REDACTED]")
        
        # Check that safe fields remain
        self.assertEqual(redacted["safe_field"], "This is safe")
        self.assertEqual(redacted["nested"]["safe_nested"], "Also safe")

    def test_access_control_validation(self) -> None:
        """Test that access control is properly validated."""
        # Test access denial
        self.mock_db.validate_access.return_value = False
        
        with self.assertRaises(PermissionError):
            self.report_generator.generate_patient_report(self.patient_id, "month")
        
        # Verify access validation was called
        self.mock_db.validate_access.assert_called_with(
            self.psychologist_id, self.patient_id, "SELECT"
        )

    def test_invalid_timeframe_handling(self) -> None:
        """Test handling of invalid timeframes."""
        with self.assertRaises(ValueError) as context:
            self.report_generator.generate_patient_report(self.patient_id, "invalid_timeframe")
        
        self.assertIn("Invalid timeframe", str(context.exception))

    @patch('psy_supabase.analytics.reports.logger')
    def test_error_logging(self, mock_logger: Mock) -> None:
        """Test that errors are properly logged."""
        # Simulate database error
        self.mock_db.get_themed_chunks_for_analysis.side_effect = Exception("Database error")
        
        report = self.report_generator.generate_patient_report(self.patient_id, "month")
        
        # Should return None on error
        self.assertIsNone(report)
        
        # Should log the error
        mock_logger.error.assert_called()

    def test_clinical_disclaimer_inclusion(self) -> None:
        """Test that clinical disclaimers are included in reports."""
        # Mock successful report generation
        self.mock_db.get_themed_chunks_for_analysis.return_value = [
            {"chunk_text": "test", "psychological_theme": "general", "embedding": [0.1] * 1536}
        ]
        self.mock_db.get_active_clinical_alerts.return_value = []
        self.mock_db.count_patient_data_points.return_value = 1
        self.mock_db.count_patient_sessions.return_value = 1
        
        with patch('psy_supabase.analytics.reports.EmotionTrajectory') as mock_trajectory:
            mock_trajectory.return_value.get_trajectory_statistics.return_value = {
                "total_data_points": 1,
                "avg_polarity": 0.0,
                "polarity_trend": "stable",
                "dominant_emotions": [],
                "concerning_episodes": 0
            }
            
            report = self.report_generator.generate_patient_report(self.patient_id, "month")
        
        self.assertIsNotNone(report)
        self.assertIn("clinical_disclaimer", report)
        self.assertIn("AI-generated", report["clinical_disclaimer"])
        self.assertIn("not a replacement for professional therapy", report["clinical_disclaimer"])


class TestReportIntegration(unittest.TestCase):
    """Integration tests for the complete reporting workflow."""

    def setUp(self) -> None:
        """Set up integration test fixtures."""
        self.mock_db = Mock(spec=DatabaseManager)
        self.psychologist_id = "psychologist_test"
        self.patient_id = "patient_test"

    @patch('psy_supabase.analytics.reports.EmotionTrajectory')
    def test_end_to_end_report_generation(self, mock_emotion_trajectory: Mock) -> None:
        """Test complete end-to-end report generation workflow."""
        # Setup comprehensive mock data
        self.mock_db.validate_access.return_value = True
        self.mock_db.get_themed_chunks_for_analysis.return_value = [
            {
                "chunk_text": "I'm feeling very anxious about my upcoming presentation at work",
                "psychological_theme": "workplace_anxiety",
                "embedding": [0.1] * 1536,
                "primary_emotion": "anxiety",
                "polarity": -0.3
            },
            {
                "chunk_text": "My relationship with my partner has been strained lately",
                "psychological_theme": "relationship_issues",
                "embedding": [0.2] * 1536,
                "primary_emotion": "sadness", 
                "polarity": -0.2
            }
        ]
        
        self.mock_db.get_active_clinical_alerts.return_value = [
            {
                "alert_id": "alert_1",
                "severity": "medium",
                "alert_type": "emotional_distress",
                "message": "Increased anxiety patterns detected",
                "created_at": datetime.now() - timedelta(hours=6),
                "metadata": {"pattern_type": "workplace_related"}
            }
        ]
        
        self.mock_db.count_patient_data_points.return_value = 25
        self.mock_db.count_patient_sessions.return_value = 5
        
        # Mock emotion trajectory
        mock_trajectory_instance = Mock()
        mock_trajectory_instance.get_trajectory_statistics.return_value = {
            "total_data_points": 25,
            "avg_polarity": -0.1,
            "polarity_std": 0.3,
            "polarity_trend": "declining",
            "dominant_emotions": ["anxiety", "sadness", "worry"],
            "concerning_episodes": 2,
            "stability_score": 0.6
        }
        mock_emotion_trajectory.return_value = mock_trajectory_instance
        
        # Generate report
        generator = ClinicalReportGenerator(self.mock_db, self.psychologist_id)
        report = generator.generate_patient_report(self.patient_id, "month")
        
        # Comprehensive assertions
        self.assertIsNotNone(report)
        
        # Verify report structure
        required_sections = [
            "patient_id", "timeframe", "generated_at", "generated_by",
            "executive_summary", "recurrent_themes", "emotional_trajectory",
            "clinical_alerts", "clinical_assessment", "recommendations",
            "clinical_disclaimer"
        ]
        
        for section in required_sections:
            self.assertIn(section, report, f"Missing required section: {section}")
        
        # Verify content quality
        self.assertEqual(report["patient_id"], self.patient_id)
        self.assertEqual(report["timeframe"], "month")
        self.assertGreater(len(report["recurrent_themes"]), 0)
        self.assertGreater(len(report["clinical_alerts"]), 0)
        self.assertIn("risk_level", report["clinical_assessment"])
        self.assertGreater(len(report["recommendations"]), 0)

    def test_role_based_access_enforcement(self) -> None:
        """Test that role-based access control is properly enforced."""
        # Test patient access denial
        with self.assertRaises(PermissionError):
            ClinicalReportGenerator(self.mock_db, "patient_123")
        
        # Test psychologist access allowed
        generator = ClinicalReportGenerator(self.mock_db, "psychologist_456")
        self.assertIsNotNone(generator)
        
        # Test admin access allowed  
        generator = ClinicalReportGenerator(self.mock_db, "admin_789")
        self.assertIsNotNone(generator)


if __name__ == "__main__":
    unittest.main()
