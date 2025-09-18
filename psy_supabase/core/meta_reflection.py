"""
Meta-reflection middleware for therapeutic chatbot metacognition.

This module implements metacognitive capabilities that allow the chatbot to reflect
on recurring patterns, emotional trajectories, and therapeutic themes across sessions.
The middleware adds professional, clinically-safe reflective statements to responses
when appropriate patterns are detected.
"""

import logging
import random
import re
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

from ..analytics.emotion_trajectory import EmotionTrajectory
from ..memory.medium_term import MediumTermMemory
from ..memory.long_term import LongTermMemory
from ..config import META_REFLECTION_CONFIG

logger = logging.getLogger(__name__)


class ReflectionType:
    """Types of metacognitive reflections."""
    THEMATIC_RECURRENCE = "thematic_recurrence"
    EMOTIONAL_TRAJECTORY = "emotional_trajectory"
    PATTERN_BRIDGING = "pattern_bridging"
    PROGRESS_ACKNOWLEDGMENT = "progress_acknowledgment"
    SAFETY_REDIRECTION = "safety_redirection"


class MetaReflectionMiddleware:
    """
    Middleware for adding metacognitive reflections to therapeutic responses.
    
    This class analyzes conversation patterns across sessions and injects
    appropriate reflective statements when recurring themes, emotional patterns,
    or concerning trajectories are detected.
    """

    def __init__(self, db_manager: "DatabaseManager", user_id: str):
        """
        Initialize meta-reflection middleware.
        
        Args:
            db_manager: Database manager instance
            user_id: Patient ID for access validation
        """
        self.db_manager = db_manager
        self.user_id = user_id
        self.emotion_tracker = EmotionTrajectory(db_manager)
        self.medium_memory = MediumTermMemory(db_manager)
        self.long_memory = LongTermMemory(db_manager)
        
        # Load configuration parameters
        self.enabled = META_REFLECTION_CONFIG.get("enabled", True)
        self.reflection_probability = META_REFLECTION_CONFIG.get("reflection_probability", 0.3)
        self.min_pattern_threshold = META_REFLECTION_CONFIG.get("min_pattern_threshold", 2)
        self.max_reflection_length = META_REFLECTION_CONFIG.get("max_reflection_length", 150)
        self.thematic_recurrence_threshold = META_REFLECTION_CONFIG.get("thematic_recurrence_threshold", 3)
        self.emotional_trajectory_days = META_REFLECTION_CONFIG.get("emotional_trajectory_days", 7)
        self.pattern_bridging_threshold = META_REFLECTION_CONFIG.get("pattern_bridging_threshold", 2)
        self.respect_clinical_alerts = META_REFLECTION_CONFIG.get("respect_clinical_alerts", True)
        self.patient_only_access = META_REFLECTION_CONFIG.get("patient_only_access", True)
        self.log_reflection_generation = META_REFLECTION_CONFIG.get("log_reflection_generation", True)
        self.log_pattern_detection = META_REFLECTION_CONFIG.get("log_pattern_detection", True)
        self.log_safety_overrides = META_REFLECTION_CONFIG.get("log_safety_overrides", True)
        
        # Initialize memory layers (remove duplicate initialization)
        # Note: self.medium_memory, self.long_memory, self.emotion_tracker already initialized above
        
        # Reflection templates
        self._load_reflection_templates()

    def _load_reflection_templates(self) -> None:
        """Load reflection templates for different pattern types."""
        self.reflection_templates = {
            ReflectionType.THEMATIC_RECURRENCE: [
                "I notice that {theme} has come up in several of our conversations. This seems to be an important area for you.",
                "We've touched on {theme} multiple times now. It appears this topic holds significant meaning in your experience.",
                "I'm observing that {theme} is a recurring theme in our sessions. This suggests it may be worth exploring further.",
                "The topic of {theme} has emerged {count} times in our recent conversations. This pattern might be worth reflecting on.",
                "I've noticed {theme} appearing consistently in our discussions. This repetition often indicates something meaningful."
            ],
            
            ReflectionType.EMOTIONAL_TRAJECTORY: [
                "I've been noticing that your emotional tone has been trending toward {direction} recently. How are you experiencing this shift?",
                "Over our recent sessions, there seems to be a pattern of {emotion_pattern}. I'm curious about your awareness of this.",
                "Your emotional expression has shown {trend_description} over time. This might be something worth acknowledging.",
                "I'm observing that your emotional state has been {stability_description} lately. How does this feel for you?",
                "There's been a noticeable {change_description} in your emotional expression across our conversations."
            ],
            
            ReflectionType.PATTERN_BRIDGING: [
                "We explored {theme} both {timeframe1} and {timeframe2}. There seems to be a connection worth exploring.",
                "I'm noticing a link between what you shared {timeframe1} about {theme} and what you're expressing now.",
                "This reminds me of something similar you mentioned {timeframe}. The pattern seems significant.",
                "We've circled back to {theme} from {timeframe}. This consistency suggests it's important to you.",
                "There's an echo here of what you shared {timeframe} about {theme}. The repetition feels meaningful."
            ],
            
            ReflectionType.PROGRESS_ACKNOWLEDGMENT: [
                "I want to acknowledge the growth I'm seeing in how you approach {theme} compared to earlier sessions.",
                "There's been a noticeable shift in your perspective on {theme} since we first discussed it.",
                "I'm observing positive changes in how you express yourself about {theme} over time.",
                "Your way of processing {theme} has evolved since our earlier conversations. That's worth recognizing.",
                "I see development in your understanding of {theme} across our sessions together."
            ],
            
            ReflectionType.SAFETY_REDIRECTION: [
                "I'm hearing some concerning themes in what you're sharing. Let's focus on what support you have available right now.",
                "What you're expressing sounds really difficult. I want to make sure you have the resources you need.",
                "I'm noticing some patterns that suggest you might benefit from additional support. How are you taking care of yourself?",
                "The intensity of what you're sharing is important. Let's talk about your safety and support systems.",
                "I want to acknowledge the pain in what you're expressing and ensure you have appropriate support."
            ]
        }

    def analyze_message(self, patient_id: str, user_message: str) -> Dict[str, Any]:
        """
        Analyze user message for patterns that warrant metacognitive reflection.
        
        Args:
            patient_id: Patient identifier
            user_message: Current user message
            
        Returns:
            Dictionary containing analysis results and reflection recommendations
        """
        if not self.enabled:
            return {"reflection_needed": False, "reason": "middleware_disabled"}
        
        try:
            analysis_result = {
                "reflection_needed": False,
                "reflection_type": None,
                "reflection_data": {},
                "safety_concerns": False,
                "patterns_detected": []
            }
            
            # Check for active clinical alerts first (safety priority)
            active_alerts = self.db_manager.get_active_clinical_alerts(patient_id)
            critical_alerts = [
                alert for alert in active_alerts 
                if alert.get("severity") == "high" and 
                alert.get("alert_type") in ["suicidal_ideation", "self_harm", "crisis"]
            ]
            
            if critical_alerts:
                analysis_result.update({
                    "reflection_needed": True,
                    "reflection_type": ReflectionType.SAFETY_REDIRECTION,
                    "safety_concerns": True,
                    "critical_alerts": critical_alerts
                })
                return analysis_result
            
            # Analyze thematic recurrence in medium-term memory
            thematic_analysis = self._analyze_thematic_recurrence(patient_id, user_message)
            if thematic_analysis["patterns_found"]:
                analysis_result["patterns_detected"].append("thematic_recurrence")
                analysis_result.update({
                    "reflection_needed": True,
                    "reflection_type": ReflectionType.THEMATIC_RECURRENCE,
                    "reflection_data": thematic_analysis
                })
            
            # Analyze emotional trajectory patterns
            if not analysis_result["reflection_needed"]:  # Only if no theme reflection
                trajectory_analysis = self._analyze_emotional_trajectory(patient_id)
                if trajectory_analysis["significant_pattern"]:
                    analysis_result["patterns_detected"].append("emotional_trajectory")
                    analysis_result.update({
                        "reflection_needed": True,
                        "reflection_type": ReflectionType.EMOTIONAL_TRAJECTORY,
                        "reflection_data": trajectory_analysis
                    })
            
            # Analyze pattern bridging opportunities
            if not analysis_result["reflection_needed"]:  # Only if no other reflection
                bridging_analysis = self._analyze_pattern_bridging(patient_id, user_message)
                if bridging_analysis["bridge_found"]:
                    analysis_result["patterns_detected"].append("pattern_bridging")
                    analysis_result.update({
                        "reflection_needed": True,
                        "reflection_type": ReflectionType.PATTERN_BRIDGING,
                        "reflection_data": bridging_analysis
                    })
            
            # Apply reflection probability
            if analysis_result["reflection_needed"] and not analysis_result["safety_concerns"]:
                if random.random() > self.reflection_probability:
                    analysis_result["reflection_needed"] = False
                    analysis_result["reason"] = "probability_filter"
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error in message analysis: {e}")
            return {"reflection_needed": False, "reason": "analysis_error", "error": str(e)}

    def _analyze_thematic_recurrence(self, patient_id: str, user_message: str) -> Dict[str, Any]:
        """Analyze for recurring therapeutic themes."""
        try:
            # Get recent medium-term memory clusters
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)  # Last 30 days
            
            medium_memories = self.medium_term_memory.retrieve(
                query=user_message,
                limit=20,
                start_date=start_date,
                end_date=end_date
            )
            
            # Extract themes from memories
            theme_counts = {}
            theme_examples = {}
            
            for memory in medium_memories:
                metadata = memory.get("metadata", {})
                theme = metadata.get("psychological_theme")
                
                if theme and theme != "general":
                    theme_counts[theme] = theme_counts.get(theme, 0) + 1
                    if theme not in theme_examples:
                        theme_examples[theme] = []
                    theme_examples[theme].append({
                        "content": memory.get("content", "")[:100],
                        "created_at": memory.get("created_at")
                    })
            
            # Find themes that meet threshold
            recurring_themes = [
                {
                    "theme": theme,
                    "count": count,
                    "examples": theme_examples[theme][:3]  # Top 3 examples
                }
                for theme, count in theme_counts.items()
                if count >= self.min_pattern_threshold
            ]
            
            # Sort by frequency
            recurring_themes.sort(key=lambda x: x["count"], reverse=True)
            
            return {
                "patterns_found": len(recurring_themes) > 0,
                "recurring_themes": recurring_themes[:2],  # Top 2 themes
                "analysis_period": "30_days",
                "total_memories_analyzed": len(medium_memories)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing thematic recurrence: {e}")
            return {"patterns_found": False, "error": str(e)}

    def _analyze_emotional_trajectory(self, patient_id: str) -> Dict[str, Any]:
        """Analyze emotional trajectory for significant patterns."""
        try:
            # Get recent trajectory statistics
            trajectory_stats = self.emotion_trajectory.get_trajectory_statistics(
                patient_id=patient_id,
                days=14  # Last 2 weeks
            )
            
            if not trajectory_stats or trajectory_stats.get("total_data_points", 0) < 5:
                return {"significant_pattern": False, "reason": "insufficient_data"}
            
            # Analyze for significant patterns
            significant_pattern = False
            pattern_description = ""
            
            # Check for concerning trends
            polarity_trend = trajectory_stats.get("polarity_trend", "stable")
            avg_polarity = trajectory_stats.get("avg_polarity", 0)
            stability_score = trajectory_stats.get("stability_score", 1.0)
            concerning_episodes = trajectory_stats.get("concerning_episodes", 0)
            
            if polarity_trend == "declining" and avg_polarity < -0.2:
                significant_pattern = True
                pattern_description = "declining emotional tone"
            elif stability_score < 0.4:
                significant_pattern = True
                pattern_description = "emotional volatility"
            elif concerning_episodes >= 2:
                significant_pattern = True
                pattern_description = "multiple concerning episodes"
            elif polarity_trend == "improving" and avg_polarity > 0.1:
                significant_pattern = True
                pattern_description = "improving emotional state"
            
            return {
                "significant_pattern": significant_pattern,
                "pattern_description": pattern_description,
                "trajectory_stats": trajectory_stats,
                "analysis_period": "14_days"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing emotional trajectory: {e}")
            return {"significant_pattern": False, "error": str(e)}

    def _analyze_pattern_bridging(self, patient_id: str, user_message: str) -> Dict[str, Any]:
        """Analyze for opportunities to bridge patterns across time."""
        try:
            # Get long-term memory for pattern bridging
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)  # Last 3 months
            
            long_memories = self.long_term_memory.retrieve(
                query=user_message,
                limit=10,
                start_date=start_date,
                end_date=end_date
            )
            
            if len(long_memories) < 2:
                return {"bridge_found": False, "reason": "insufficient_historical_data"}
            
            # Look for thematic connections across time
            current_themes = self._extract_themes_from_message(user_message)
            
            for memory in long_memories:
                memory_metadata = memory.get("metadata", {})
                memory_theme = memory_metadata.get("psychological_theme")
                memory_date = memory.get("created_at")
                
                if memory_theme in current_themes and memory_date:
                    # Calculate time difference
                    if isinstance(memory_date, str):
                        memory_date = datetime.fromisoformat(memory_date.replace('Z', '+00:00'))
                    
                    days_ago = (datetime.now() - memory_date).days
                    
                    if days_ago >= 7:  # At least a week ago
                        timeframe = self._format_timeframe(days_ago)
                        
                        return {
                            "bridge_found": True,
                            "theme": memory_theme,
                            "timeframe": timeframe,
                            "historical_content": memory.get("content", "")[:150],
                            "days_ago": days_ago
                        }
            
            return {"bridge_found": False, "reason": "no_temporal_patterns"}
            
        except Exception as e:
            logger.error(f"Error analyzing pattern bridging: {e}")
            return {"bridge_found": False, "error": str(e)}

    def _extract_themes_from_message(self, message: str) -> List[str]:
        """Extract potential psychological themes from a message."""
        theme_keywords = {
            "workplace_anxiety": ["work", "job", "boss", "colleague", "deadline", "office", "career"],
            "relationship_issues": ["relationship", "partner", "marriage", "dating", "love", "breakup", "family"],
            "self_esteem": ["confidence", "self-worth", "insecure", "doubt", "worthless", "shame"],
            "general_anxiety": ["anxiety", "worry", "nervous", "panic", "fear", "stress"],
            "depression": ["sad", "depressed", "hopeless", "empty", "lonely", "down"],
            "grief_loss": ["loss", "grief", "death", "died", "mourning", "goodbye"],
            "trauma": ["trauma", "flashback", "triggered", "ptsd", "abuse"],
            "addiction": ["addiction", "substance", "drinking", "drugs", "recovery"]
        }
        
        message_lower = message.lower()
        detected_themes = []
        
        for theme, keywords in theme_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                detected_themes.append(theme)
        
        return detected_themes

    def _format_timeframe(self, days_ago: int) -> str:
        """Format timeframe in human-readable form."""
        if days_ago < 14:
            return f"{days_ago} days ago"
        elif days_ago < 30:
            weeks = days_ago // 7
            return f"{weeks} week{'s' if weeks > 1 else ''} ago"
        elif days_ago < 90:
            months = days_ago // 30
            return f"{months} month{'s' if months > 1 else ''} ago"
        else:
            return "several months ago"

    def inject_meta_reflection(
        self,
        response: str,
        analysis_result: Dict[str, Any]
    ) -> str:
        """
        Inject metacognitive reflection into the response.
        
        Args:
            response: Original chatbot response
            analysis_result: Result from analyze_message
            
        Returns:
            Response with optional reflection injected
        """
        if not analysis_result.get("reflection_needed", False):
            return response
        
        try:
            reflection_type = analysis_result.get("reflection_type")
            reflection_data = analysis_result.get("reflection_data", {})
            
            reflection_text = self._generate_reflection(reflection_type, reflection_data)
            
            if reflection_text:
                # Insert reflection at appropriate point in response
                if analysis_result.get("safety_concerns", False):
                    # For safety concerns, replace or prepend
                    return f"{reflection_text}\n\n{response}"
                else:
                    # For other reflections, append thoughtfully
                    return f"{response}\n\n{reflection_text}"
            
            return response
            
        except Exception as e:
            logger.error(f"Error injecting reflection: {e}")
            return response

    def _generate_reflection(self, reflection_type: str, reflection_data: Dict[str, Any]) -> str:
        """Generate reflection text based on type and data."""
        try:
            templates = self.reflection_templates.get(reflection_type, [])
            if not templates:
                return ""
            
            template = random.choice(templates)
            
            if reflection_type == ReflectionType.THEMATIC_RECURRENCE:
                themes = reflection_data.get("recurring_themes", [])
                if themes:
                    top_theme = themes[0]
                    return template.format(
                        theme=self._humanize_theme(top_theme["theme"]),
                        count=top_theme["count"]
                    )
            
            elif reflection_type == ReflectionType.EMOTIONAL_TRAJECTORY:
                pattern_desc = reflection_data.get("pattern_description", "")
                trajectory_stats = reflection_data.get("trajectory_stats", {})
                
                if pattern_desc:
                    return template.format(
                        direction=pattern_desc,
                        emotion_pattern=pattern_desc,
                        trend_description=pattern_desc,
                        stability_description=pattern_desc,
                        change_description=pattern_desc
                    )
            
            elif reflection_type == ReflectionType.PATTERN_BRIDGING:
                theme = reflection_data.get("theme", "")
                timeframe = reflection_data.get("timeframe", "")
                
                if theme and timeframe:
                    return template.format(
                        theme=self._humanize_theme(theme),
                        timeframe=timeframe,
                        timeframe1=timeframe,
                        timeframe2="now"
                    )
            
            elif reflection_type == ReflectionType.SAFETY_REDIRECTION:
                return template  # Safety templates don't need formatting
            
            return ""
            
        except Exception as e:
            logger.error(f"Error generating reflection text: {e}")
            return ""

    def _humanize_theme(self, theme: str) -> str:
        """Convert technical theme names to human-readable form."""
        theme_map = {
            "workplace_anxiety": "work-related stress",
            "relationship_issues": "relationship concerns",
            "self_esteem": "self-worth and confidence",
            "general_anxiety": "anxiety",
            "depression": "feelings of sadness",
            "grief_loss": "loss and grief",
            "trauma": "difficult past experiences",
            "addiction": "substance use concerns"
        }
        return theme_map.get(theme, theme.replace("_", " "))

    def trigger_reflection_on_alerts(self, patient_id: str) -> Optional[str]:
        """
        Generate reflection when clinical alerts are triggered.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            Reflection text if appropriate, None otherwise
        """
        try:
            active_alerts = self.db_manager.get_active_clinical_alerts(patient_id)
            
            if not active_alerts:
                return None
            
            # Check for critical alerts that require safety redirection
            critical_alerts = [
                alert for alert in active_alerts
                if alert.get("severity") == "high" and
                alert.get("alert_type") in ["suicidal_ideation", "self_harm", "crisis"]
            ]
            
            if critical_alerts:
                templates = self.reflection_templates[ReflectionType.SAFETY_REDIRECTION]
                return random.choice(templates)
            
            # For non-critical alerts, generate supportive reflection
            high_severity_alerts = [
                alert for alert in active_alerts
                if alert.get("severity") == "high"
            ]
            
            if high_severity_alerts:
                return ("I'm noticing some patterns in our conversations that suggest "
                       "you might be going through a particularly challenging time. "
                       "I want to acknowledge that and ensure you have the support you need.")
            
            return None
            
        except Exception as e:
            logger.error(f"Error triggering reflection on alerts: {e}")
            return None

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable the middleware."""
        self.enabled = enabled
        logger.info(f"Meta-reflection middleware {'enabled' if enabled else 'disabled'}")

    def get_reflection_stats(self, patient_id: str) -> Dict[str, Any]:
        """Get statistics about reflections for a patient."""
        try:
            # This would typically be stored in database, for now return basic info
            return {
                "middleware_enabled": self.enabled,
                "reflection_probability": self.reflection_probability,
                "min_pattern_threshold": self.min_pattern_threshold,
                "available_reflection_types": list(self.reflection_templates.keys()),
                "patient_id": patient_id
            }
        except Exception as e:
            logger.error(f"Error getting reflection stats: {e}")
            return {"error": str(e)}
