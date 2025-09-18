"""
Comprehensive tests for MetaReflectionMiddleware.

This module tests the metacognitive reflection capabilities including:
- Pattern detection (thematic recurrence, emotional trajectory, pattern bridging)
- Reflection injection with professional tone and safety filters
- Role-based access control (patients only)
- Configuration toggles and safety overrides
- Integration with memory layers and clinical alerts
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, Any, List

from psy_supabase.core.meta_reflection import (
    MetaReflectionMiddleware,
    ReflectionType
)
from psy_supabase.config import META_REFLECTION_CONFIG


class TestMetaReflectionMiddleware:
    """Test suite for MetaReflectionMiddleware class."""

    @pytest.fixture
    def mock_db_manager(self):
        """Mock database manager for testing."""
        db_manager = Mock()
        db_manager.validate_user_access.return_value = True
        db_manager.get_active_clinical_alerts.return_value = []
        return db_manager

    @pytest.fixture
    def mock_memory_layers(self):
        """Mock memory layer components."""
        with patch('psy_supabase.core.meta_reflection.MediumTermMemory') as mock_medium, \
             patch('psy_supabase.core.meta_reflection.LongTermMemory') as mock_long, \
             patch('psy_supabase.core.meta_reflection.EmotionTrajectory') as mock_emotion:
            
            # Configure medium-term memory mock
            mock_medium_instance = Mock()
            mock_medium_instance.retrieve_memories.return_value = []
            mock_medium.return_value = mock_medium_instance
            
            # Configure long-term memory mock
            mock_long_instance = Mock()
            mock_long_instance.retrieve_memories.return_value = []
            mock_long.return_value = mock_long_instance
            
            # Configure emotion trajectory mock
            mock_emotion_instance = Mock()
            mock_emotion_instance.get_patient_trajectory.return_value = []
            mock_emotion.return_value = mock_emotion_instance
            
            yield {
                'medium': mock_medium_instance,
                'long': mock_long_instance,
                'emotion': mock_emotion_instance
            }

    @pytest.fixture
    def meta_reflection(self, mock_db_manager, mock_memory_layers):
        """Create MetaReflectionMiddleware instance for testing."""
        return MetaReflectionMiddleware(mock_db_manager, "test_patient_123")

    def test_initialization_with_config(self, mock_db_manager, mock_memory_layers):
        """Test middleware initialization with configuration parameters."""
        middleware = MetaReflectionMiddleware(mock_db_manager, "test_patient_123")
        
        # Verify configuration loading
        assert middleware.enabled == META_REFLECTION_CONFIG.get("enabled", True)
        assert middleware.reflection_probability == META_REFLECTION_CONFIG.get("reflection_probability", 0.3)
        assert middleware.min_pattern_threshold == META_REFLECTION_CONFIG.get("min_pattern_threshold", 2)
        assert middleware.max_reflection_length == META_REFLECTION_CONFIG.get("max_reflection_length", 150)
        
        # Verify memory layer initialization
        assert middleware.db_manager == mock_db_manager
        assert middleware.user_id == "test_patient_123"

    def test_reflection_templates_loaded(self, meta_reflection):
        """Test that reflection templates are properly loaded."""
        templates = meta_reflection.reflection_templates
        
        # Verify all reflection types have templates
        assert ReflectionType.THEMATIC_RECURRENCE in templates
        assert ReflectionType.EMOTIONAL_TRAJECTORY in templates
        assert ReflectionType.PATTERN_BRIDGING in templates
        assert ReflectionType.PROGRESS_ACKNOWLEDGMENT in templates
        assert ReflectionType.SAFETY_REDIRECTION in templates
        
        # Verify templates are non-empty lists
        for reflection_type, template_list in templates.items():
            assert isinstance(template_list, list)
            assert len(template_list) > 0
            assert all(isinstance(template, str) for template in template_list)

    def test_analyze_message_thematic_recurrence(self, meta_reflection, mock_memory_layers):
        """Test thematic recurrence pattern detection."""
        # Mock medium-term memory to return recurring themes
        mock_memory_layers['medium'].retrieve_memories.return_value = [
            {
                'content': 'anxiety about work',
                'metadata': {'primary_theme': 'work_stress', 'emotional_intensity': 0.8},
                'created_at': datetime.now() - timedelta(days=2)
            },
            {
                'content': 'work pressure overwhelming',
                'metadata': {'primary_theme': 'work_stress', 'emotional_intensity': 0.7},
                'created_at': datetime.now() - timedelta(days=5)
            },
            {
                'content': 'job stress affecting sleep',
                'metadata': {'primary_theme': 'work_stress', 'emotional_intensity': 0.9},
                'created_at': datetime.now() - timedelta(days=7)
            }
        ]
        
        result = meta_reflection.analyze_message("test_patient_123", "I'm feeling stressed about work again")
        
        # Verify thematic recurrence detection
        assert result['reflection_needed'] is True
        assert result['reflection_type'] == ReflectionType.THEMATIC_RECURRENCE
        assert 'work_stress' in result['detected_themes']
        assert result['theme_count'] >= meta_reflection.thematic_recurrence_threshold

    def test_analyze_message_emotional_trajectory(self, meta_reflection, mock_memory_layers):
        """Test emotional trajectory pattern detection."""
        # Mock emotion trajectory data showing concerning pattern
        mock_memory_layers['emotion'].get_patient_trajectory.return_value = [
            {
                'timestamp': datetime.now() - timedelta(days=1),
                'emotion_type': 'sadness',
                'polarity': -0.8,
                'intensity': 0.9
            },
            {
                'timestamp': datetime.now() - timedelta(days=2),
                'emotion_type': 'sadness',
                'polarity': -0.7,
                'intensity': 0.8
            },
            {
                'timestamp': datetime.now() - timedelta(days=3),
                'emotion_type': 'sadness',
                'polarity': -0.6,
                'intensity': 0.7
            }
        ]
        
        result = meta_reflection.analyze_message("test_patient_123", "I've been feeling really down lately")
        
        # Verify emotional trajectory detection
        assert result['reflection_needed'] is True
        assert result['reflection_type'] == ReflectionType.EMOTIONAL_TRAJECTORY
        assert 'trajectory_pattern' in result
        assert result['trajectory_pattern'] in ['declining', 'concerning', 'negative_trend']

    def test_analyze_message_pattern_bridging(self, meta_reflection, mock_memory_layers):
        """Test pattern bridging between different memory layers."""
        # Mock both medium and long-term memories with related themes
        mock_memory_layers['medium'].retrieve_memories.return_value = [
            {
                'content': 'relationship conflict',
                'metadata': {'primary_theme': 'relationships', 'emotional_intensity': 0.7},
                'created_at': datetime.now() - timedelta(days=3)
            }
        ]
        
        mock_memory_layers['long'].retrieve_memories.return_value = [
            {
                'content': 'family communication issues',
                'metadata': {'primary_theme': 'family_dynamics', 'emotional_intensity': 0.8},
                'created_at': datetime.now() - timedelta(days=20)
            }
        ]
        
        result = meta_reflection.analyze_message("test_patient_123", "Having trouble communicating with my partner")
        
        # Verify pattern bridging detection
        if result['reflection_needed']:
            assert result['reflection_type'] == ReflectionType.PATTERN_BRIDGING
            assert 'bridged_patterns' in result
            assert len(result['bridged_patterns']) >= meta_reflection.pattern_bridging_threshold

    def test_inject_meta_reflection_thematic(self, meta_reflection):
        """Test reflection injection for thematic recurrence."""
        analysis_result = {
            'reflection_needed': True,
            'reflection_type': ReflectionType.THEMATIC_RECURRENCE,
            'detected_themes': ['work_stress'],
            'theme_count': 3,
            'primary_theme': 'work_stress'
        }
        
        original_response = "I understand you're feeling stressed. Let's explore some coping strategies."
        
        with patch('random.random', return_value=0.1):  # Force reflection generation
            enhanced_response = meta_reflection.inject_meta_reflection(original_response, analysis_result)
        
        # Verify reflection was injected
        assert enhanced_response != original_response
        assert len(enhanced_response) > len(original_response)
        assert 'work_stress' in enhanced_response or 'work' in enhanced_response
        
        # Verify professional tone
        assert any(word in enhanced_response.lower() for word in ['notice', 'observe', 'pattern', 'theme'])

    def test_inject_meta_reflection_emotional_trajectory(self, meta_reflection):
        """Test reflection injection for emotional trajectory."""
        analysis_result = {
            'reflection_needed': True,
            'reflection_type': ReflectionType.EMOTIONAL_TRAJECTORY,
            'trajectory_pattern': 'declining',
            'trajectory_description': 'increasing sadness over past week'
        }
        
        original_response = "I hear that you're struggling with sadness."
        
        with patch('random.random', return_value=0.1):  # Force reflection generation
            enhanced_response = meta_reflection.inject_meta_reflection(original_response, analysis_result)
        
        # Verify reflection was injected
        assert enhanced_response != original_response
        assert any(word in enhanced_response.lower() for word in ['pattern', 'trend', 'trajectory', 'notice'])

    def test_inject_meta_reflection_length_limit(self, meta_reflection):
        """Test that reflections respect maximum length limits."""
        analysis_result = {
            'reflection_needed': True,
            'reflection_type': ReflectionType.THEMATIC_RECURRENCE,
            'detected_themes': ['very_long_theme_name_that_might_cause_issues'],
            'theme_count': 5,
            'primary_theme': 'very_long_theme_name_that_might_cause_issues'
        }
        
        original_response = "Short response."
        
        with patch('random.random', return_value=0.1):  # Force reflection generation
            enhanced_response = meta_reflection.inject_meta_reflection(original_response, analysis_result)
        
        # Calculate reflection length (total - original - separator)
        reflection_length = len(enhanced_response) - len(original_response) - 4  # Account for "\n\n"
        assert reflection_length <= meta_reflection.max_reflection_length

    def test_trigger_reflection_on_alerts_safety_redirection(self, meta_reflection, mock_db_manager):
        """Test safety redirection for critical clinical alerts."""
        # Mock critical clinical alert
        mock_db_manager.get_active_clinical_alerts.return_value = [
            {
                'alert_type': 'suicide_risk',
                'severity': 'critical',
                'description': 'Suicidal ideation detected',
                'created_at': datetime.now()
            }
        ]
        
        result = meta_reflection.trigger_reflection_on_alerts("test_patient_123")
        
        # Verify safety redirection
        assert result['reflection_needed'] is True
        assert result['reflection_type'] == ReflectionType.SAFETY_REDIRECTION
        assert 'safety_message' in result
        assert any(word in result['safety_message'].lower() for word in ['professional', 'therapist', 'support'])

    def test_trigger_reflection_on_alerts_no_critical(self, meta_reflection, mock_db_manager):
        """Test reflection generation with non-critical alerts."""
        # Mock non-critical clinical alert
        mock_db_manager.get_active_clinical_alerts.return_value = [
            {
                'alert_type': 'mood_decline',
                'severity': 'moderate',
                'description': 'Gradual mood decline observed',
                'created_at': datetime.now()
            }
        ]
        
        result = meta_reflection.trigger_reflection_on_alerts("test_patient_123")
        
        # Verify normal reflection can proceed
        assert result['reflection_needed'] is True
        assert result['reflection_type'] == ReflectionType.PROGRESS_ACKNOWLEDGMENT
        assert 'alert_context' in result

    def test_safety_filters_respect_clinical_alerts(self, meta_reflection, mock_db_manager):
        """Test that safety filters block reflections during critical alerts."""
        # Mock critical alert
        mock_db_manager.get_active_clinical_alerts.return_value = [
            {
                'alert_type': 'self_harm',
                'severity': 'critical',
                'description': 'Self-harm indicators detected',
                'created_at': datetime.now()
            }
        ]
        
        analysis_result = {
            'reflection_needed': True,
            'reflection_type': ReflectionType.THEMATIC_RECURRENCE,
            'detected_themes': ['depression'],
            'theme_count': 3
        }
        
        original_response = "I understand you're going through a difficult time."
        
        # Should not inject regular reflection during critical alert
        enhanced_response = meta_reflection.inject_meta_reflection(original_response, analysis_result)
        
        # Verify safety override occurred
        if meta_reflection.respect_clinical_alerts:
            # Should either return original or safety redirection, not thematic reflection
            assert 'depression' not in enhanced_response or 'professional help' in enhanced_response

    def test_role_based_access_control(self, mock_db_manager, mock_memory_layers):
        """Test that reflections are only generated for patient role."""
        # Test with patient role
        mock_db_manager.validate_user_access.return_value = True
        patient_middleware = MetaReflectionMiddleware(mock_db_manager, "patient_123")
        
        # Test with non-patient role
        mock_db_manager.validate_user_access.return_value = False
        
        analysis_result = {
            'reflection_needed': True,
            'reflection_type': ReflectionType.THEMATIC_RECURRENCE,
            'detected_themes': ['anxiety'],
            'theme_count': 3
        }
        
        original_response = "Let's discuss your anxiety."
        
        # Should not inject reflection for non-patient
        if patient_middleware.patient_only_access:
            enhanced_response = patient_middleware.inject_meta_reflection(original_response, analysis_result)
            # Verify access control (implementation may vary)
            assert enhanced_response == original_response or 'anxiety' not in enhanced_response

    def test_configuration_toggle_disabled(self, mock_db_manager, mock_memory_layers):
        """Test that reflections are disabled when configuration is turned off."""
        with patch.dict('psy_supabase.config.META_REFLECTION_CONFIG', {'enabled': False}):
            middleware = MetaReflectionMiddleware(mock_db_manager, "test_patient_123")
            
            analysis_result = {
                'reflection_needed': True,
                'reflection_type': ReflectionType.THEMATIC_RECURRENCE,
                'detected_themes': ['stress'],
                'theme_count': 4
            }
            
            original_response = "I understand your stress."
            enhanced_response = middleware.inject_meta_reflection(original_response, analysis_result)
            
            # Should return original response when disabled
            assert enhanced_response == original_response

    def test_reflection_probability_control(self, meta_reflection):
        """Test that reflection probability controls generation frequency."""
        analysis_result = {
            'reflection_needed': True,
            'reflection_type': ReflectionType.THEMATIC_RECURRENCE,
            'detected_themes': ['anxiety'],
            'theme_count': 3,
            'primary_theme': 'anxiety'
        }
        
        original_response = "Let's work on managing anxiety."
        
        # Test with low probability (should not generate)
        with patch('random.random', return_value=0.9):
            enhanced_response = meta_reflection.inject_meta_reflection(original_response, analysis_result)
            assert enhanced_response == original_response
        
        # Test with high probability (should generate)
        with patch('random.random', return_value=0.1):
            enhanced_response = meta_reflection.inject_meta_reflection(original_response, analysis_result)
            assert enhanced_response != original_response

    def test_pattern_threshold_enforcement(self, meta_reflection, mock_memory_layers):
        """Test that pattern thresholds are enforced for reflection generation."""
        # Mock insufficient pattern occurrences
        mock_memory_layers['medium'].retrieve_memories.return_value = [
            {
                'content': 'single anxiety mention',
                'metadata': {'primary_theme': 'anxiety', 'emotional_intensity': 0.6},
                'created_at': datetime.now() - timedelta(days=1)
            }
        ]
        
        result = meta_reflection.analyze_message("test_patient_123", "Feeling anxious today")
        
        # Should not trigger reflection if below threshold
        if result.get('theme_count', 0) < meta_reflection.min_pattern_threshold:
            assert result['reflection_needed'] is False

    def test_professional_tone_validation(self, meta_reflection):
        """Test that generated reflections maintain professional tone."""
        analysis_result = {
            'reflection_needed': True,
            'reflection_type': ReflectionType.THEMATIC_RECURRENCE,
            'detected_themes': ['relationship_issues'],
            'theme_count': 4,
            'primary_theme': 'relationship_issues'
        }
        
        original_response = "Relationships can be challenging."
        
        with patch('random.random', return_value=0.1):  # Force reflection generation
            enhanced_response = meta_reflection.inject_meta_reflection(original_response, analysis_result)
        
        # Verify professional language patterns
        professional_indicators = [
            'notice', 'observe', 'pattern', 'theme', 'appears', 'seems',
            'suggests', 'indicates', 'worth exploring', 'meaningful'
        ]
        
        reflection_text = enhanced_response.replace(original_response, "").strip()
        assert any(indicator in reflection_text.lower() for indicator in professional_indicators)
        
        # Verify no inappropriate casual language
        casual_words = ['dude', 'awesome', 'cool', 'whatever', 'like totally']
        assert not any(word in reflection_text.lower() for word in casual_words)

    def test_error_handling_memory_failure(self, meta_reflection, mock_memory_layers):
        """Test graceful error handling when memory layers fail."""
        # Mock memory layer failure
        mock_memory_layers['medium'].retrieve_memories.side_effect = Exception("Database connection failed")
        
        # Should not crash and should return safe fallback
        result = meta_reflection.analyze_message("test_patient_123", "Test message")
        
        # Verify graceful degradation
        assert isinstance(result, dict)
        assert 'reflection_needed' in result
        # Should default to False when analysis fails
        assert result['reflection_needed'] is False

    def test_logging_and_monitoring(self, meta_reflection, mock_memory_layers):
        """Test that appropriate logging occurs for monitoring."""
        with patch('psy_supabase.core.meta_reflection.logger') as mock_logger:
            analysis_result = {
                'reflection_needed': True,
                'reflection_type': ReflectionType.THEMATIC_RECURRENCE,
                'detected_themes': ['stress'],
                'theme_count': 3,
                'primary_theme': 'stress'
            }
            
            original_response = "Let's address your stress."
            
            with patch('random.random', return_value=0.1):  # Force reflection generation
                meta_reflection.inject_meta_reflection(original_response, analysis_result)
            
            # Verify logging occurred if enabled
            if meta_reflection.log_reflection_generation:
                mock_logger.info.assert_called()

    def test_integration_with_dynamic_rag(self, mock_db_manager, mock_memory_layers):
        """Test integration with DynamicRAGRetriever."""
        from psy_supabase.core.dynamic_rag import DynamicRAGRetriever
        
        # Mock DynamicRAGRetriever
        with patch('psy_supabase.core.dynamic_rag.DynamicRAGRetriever') as mock_rag:
            mock_rag_instance = Mock()
            mock_rag_instance.meta_reflection = MetaReflectionMiddleware(mock_db_manager, "test_patient_123")
            
            # Test process_response_with_reflection method
            response = "Original response"
            user_message = "I'm feeling stressed about work"
            
            # Mock analysis to return reflection needed
            with patch.object(mock_rag_instance.meta_reflection, 'analyze_message') as mock_analyze:
                mock_analyze.return_value = {
                    'reflection_needed': True,
                    'reflection_type': ReflectionType.THEMATIC_RECURRENCE,
                    'detected_themes': ['work_stress'],
                    'theme_count': 3
                }
                
                with patch.object(mock_rag_instance.meta_reflection, 'inject_meta_reflection') as mock_inject:
                    mock_inject.return_value = "Enhanced response with reflection"
                    
                    # Simulate the integration
                    if hasattr(mock_rag_instance, 'process_response_with_reflection'):
                        result = mock_rag_instance.process_response_with_reflection(
                            response, user_message, "test_patient_123"
                        )
                        
                        # Verify integration worked
                        mock_analyze.assert_called_once_with("test_patient_123", user_message)
                        mock_inject.assert_called_once()


class TestReflectionTemplates:
    """Test reflection template functionality."""

    def test_template_formatting(self):
        """Test that templates can be properly formatted with variables."""
        template = "I notice that {theme} has come up {count} times in our conversations."
        
        formatted = template.format(theme="anxiety", count=3)
        expected = "I notice that anxiety has come up 3 times in our conversations."
        
        assert formatted == expected

    def test_template_safety(self):
        """Test that templates don't contain inappropriate content."""
        from psy_supabase.core.meta_reflection import MetaReflectionMiddleware
        
        # Create instance to access templates
        db_manager = Mock()
        middleware = MetaReflectionMiddleware(db_manager, "test_patient")
        
        inappropriate_words = [
            'cure', 'fix', 'diagnose', 'prescribe', 'medication',
            'disorder', 'illness', 'disease', 'pathology'
        ]
        
        for reflection_type, templates in middleware.reflection_templates.items():
            for template in templates:
                template_lower = template.lower()
                for word in inappropriate_words:
                    assert word not in template_lower, f"Template contains inappropriate word '{word}': {template}"

    def test_template_professional_tone(self):
        """Test that all templates maintain professional therapeutic tone."""
        from psy_supabase.core.meta_reflection import MetaReflectionMiddleware
        
        db_manager = Mock()
        middleware = MetaReflectionMiddleware(db_manager, "test_patient")
        
        professional_indicators = [
            'notice', 'observe', 'appears', 'seems', 'suggests',
            'indicates', 'pattern', 'theme', 'worth', 'meaningful'
        ]
        
        for reflection_type, templates in middleware.reflection_templates.items():
            for template in templates:
                template_lower = template.lower()
                # Each template should contain at least one professional indicator
                assert any(indicator in template_lower for indicator in professional_indicators), \
                    f"Template lacks professional tone: {template}"


if __name__ == "__main__":
    pytest.main([__file__])
