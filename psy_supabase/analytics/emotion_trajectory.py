"""
Emotional Trajectory Tracking and Analysis for Psychobot Therapeutic System.

This module provides comprehensive emotional trajectory tracking, analysis, and visualization
for therapeutic sessions. It enables clinicians to monitor patient emotional patterns over time,
detect concerning changes, and generate clinical insights.

Key Features:
- Real-time emotional state tracking from session chunks
- Time-series analysis of emotional patterns
- Abrupt change detection for clinical alerts
- Interactive visualization with matplotlib/seaborn
- Clinical safety monitoring with automated alerts
- Multi-timeframe analysis (sessions, weeks, months)

Clinical Applications:
- Treatment progress monitoring
- Early intervention triggers
- Therapeutic approach effectiveness
- Patient emotional stability assessment
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats

from psy_supabase import get_package_logger
from psy_supabase.core.database import DatabaseManager
from psy_supabase.db.policies import UserRole

logger = get_package_logger(__name__)


class EmotionType(Enum):
    """Enumeration of primary emotion types."""
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEAR = "fear"
    NEUTRAL = "neutral"
    ANXIETY = "anxiety"
    DEPRESSION = "depression"


class AlertSeverity(Enum):
    """Enumeration of clinical alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class EmotionalDataPoint:
    """Represents a single emotional data point in time."""
    timestamp: datetime
    emotion: EmotionType
    intensity: float  # 0.0 to 1.0
    polarity: float   # -1.0 to 1.0
    session_id: str
    chunk_id: Optional[str] = None
    theme: Optional[str] = None


@dataclass
class TrajectoryAlert:
    """Represents a clinical alert for emotional trajectory changes."""
    patient_id: str
    alert_type: str
    severity: AlertSeverity
    message: str
    timestamp: datetime
    data_points: List[EmotionalDataPoint]
    threshold_exceeded: float


class EmotionTrajectory:
    """
    Manages emotional trajectory tracking and analysis for therapeutic sessions.
    
    This class provides methods to record emotional states, analyze patterns over time,
    detect concerning changes, and generate visualizations for clinical use.
    """

    def __init__(self, database_manager: DatabaseManager):
        """
        Initialize the emotion trajectory tracker.
        
        Args:
            database_manager: DatabaseManager instance for data persistence
        """
        self.db_manager = database_manager
        self._emotion_cache: Dict[str, List[EmotionalDataPoint]] = {}
        
        # Configure matplotlib for clinical visualizations
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def record_chunk_emotion(
        self, 
        patient_id: str,
        session_id: str,
        chunk_metadata: Dict[str, Any],
        chunk_id: Optional[str] = None
    ) -> bool:
        """
        Record emotional data from a session chunk.
        
        Args:
            patient_id: Patient identifier
            session_id: Session identifier
            chunk_metadata: Metadata containing emotion, polarity, intensity, theme
            chunk_id: Optional chunk identifier
            
        Returns:
            bool: True if recording was successful
        """
        try:
            # Extract emotional data from metadata
            emotion_str = chunk_metadata.get("primary_emotion", "neutral")
            intensity = float(chunk_metadata.get("intensity", 0.0))
            polarity = float(chunk_metadata.get("polarity", 0.0))
            theme = chunk_metadata.get("theme")
            
            # Validate emotion type
            try:
                emotion = EmotionType(emotion_str.lower())
            except ValueError:
                logger.warning(f"Unknown emotion type: {emotion_str}, defaulting to neutral")
                emotion = EmotionType.NEUTRAL
            
            # Validate intensity and polarity ranges
            intensity = max(0.0, min(1.0, intensity))
            polarity = max(-1.0, min(1.0, polarity))
            
            # Create emotional data point
            data_point = EmotionalDataPoint(
                timestamp=datetime.now(),
                emotion=emotion,
                intensity=intensity,
                polarity=polarity,
                session_id=session_id,
                chunk_id=chunk_id,
                theme=theme
            )
            
            # Store in database
            success = self._persist_emotional_datapoint(patient_id, data_point)
            
            if success:
                # Update cache
                if patient_id not in self._emotion_cache:
                    self._emotion_cache[patient_id] = []
                self._emotion_cache[patient_id].append(data_point)
                
                # Check for abrupt changes
                self._check_abrupt_changes(patient_id)
                
                logger.info(f"Recorded emotion {emotion.value} (intensity: {intensity:.2f}, polarity: {polarity:.2f}) for patient {patient_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error recording chunk emotion: {e}")
            return False

    def get_patient_trajectory(
        self, 
        patient_id: str, 
        timeframe: str = "1_month"
    ) -> List[EmotionalDataPoint]:
        """
        Get emotional trajectory for a patient within a specified timeframe.
        
        Args:
            patient_id: Patient identifier
            timeframe: Time period ("1_week", "1_month", "3_months", "6_months", "1_year")
            
        Returns:
            List of EmotionalDataPoint objects ordered by timestamp
        """
        try:
            # Calculate time range
            end_date = datetime.now()
            timeframe_map = {
                "1_week": timedelta(weeks=1),
                "1_month": timedelta(days=30),
                "3_months": timedelta(days=90),
                "6_months": timedelta(days=180),
                "1_year": timedelta(days=365)
            }
            
            if timeframe not in timeframe_map:
                logger.warning(f"Unknown timeframe: {timeframe}, defaulting to 1_month")
                timeframe = "1_month"
            
            start_date = end_date - timeframe_map[timeframe]
            
            # Retrieve from database
            trajectory_data = self._retrieve_trajectory_data(patient_id, start_date, end_date)
            
            # Sort by timestamp
            trajectory_data.sort(key=lambda x: x.timestamp)
            
            logger.info(f"Retrieved {len(trajectory_data)} emotional data points for patient {patient_id} in timeframe {timeframe}")
            return trajectory_data
            
        except Exception as e:
            logger.error(f"Error getting patient trajectory: {e}")
            return []

    def detect_abrupt_changes(
        self, 
        patient_id: str, 
        threshold: float = 0.3,
        window_size: int = 5
    ) -> List[TrajectoryAlert]:
        """
        Detect abrupt changes in emotional trajectory.
        
        Args:
            patient_id: Patient identifier
            threshold: Minimum change threshold (0.0 to 1.0)
            window_size: Number of recent data points to analyze
            
        Returns:
            List of TrajectoryAlert objects for concerning changes
        """
        try:
            alerts = []
            
            # Get recent trajectory data
            recent_data = self.get_patient_trajectory(patient_id, "1_week")
            
            if len(recent_data) < window_size:
                logger.debug(f"Insufficient data for abrupt change detection: {len(recent_data)} points")
                return alerts
            
            # Analyze recent window
            recent_window = recent_data[-window_size:]
            
            # Calculate polarity changes
            polarities = [dp.polarity for dp in recent_window]
            polarity_change = max(polarities) - min(polarities)
            
            # Check for significant negative polarity increase
            if polarity_change > threshold:
                recent_avg = np.mean(polarities[-3:]) if len(polarities) >= 3 else polarities[-1]
                baseline_avg = np.mean(polarities[:-3]) if len(polarities) >= 6 else 0.0
                
                if recent_avg < baseline_avg - threshold:
                    severity = AlertSeverity.HIGH if polarity_change > 0.5 else AlertSeverity.MEDIUM
                    
                    alert = TrajectoryAlert(
                        patient_id=patient_id,
                        alert_type="abrupt_negative_change",
                        severity=severity,
                        message=f"Detected {polarity_change:.2f} decrease in emotional polarity over recent sessions",
                        timestamp=datetime.now(),
                        data_points=recent_window,
                        threshold_exceeded=polarity_change
                    )
                    alerts.append(alert)
            
            # Check for sustained negative emotions
            negative_count = sum(1 for dp in recent_window if dp.polarity < -0.3)
            if negative_count >= window_size * 0.7:  # 70% of recent sessions
                alert = TrajectoryAlert(
                    patient_id=patient_id,
                    alert_type="sustained_negative_emotion",
                    severity=AlertSeverity.HIGH,
                    message=f"Sustained negative emotions detected in {negative_count}/{window_size} recent sessions",
                    timestamp=datetime.now(),
                    data_points=recent_window,
                    threshold_exceeded=negative_count / window_size
                )
                alerts.append(alert)
            
            # Check for emotional intensity spikes
            intensities = [dp.intensity for dp in recent_window]
            intensity_std = np.std(intensities)
            if intensity_std > 0.4:  # High variability
                alert = TrajectoryAlert(
                    patient_id=patient_id,
                    alert_type="emotional_instability",
                    severity=AlertSeverity.MEDIUM,
                    message=f"High emotional variability detected (std: {intensity_std:.2f})",
                    timestamp=datetime.now(),
                    data_points=recent_window,
                    threshold_exceeded=intensity_std
                )
                alerts.append(alert)
            
            # Log clinical alerts
            for alert in alerts:
                self._log_clinical_alert(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error detecting abrupt changes: {e}")
            return []

    def export_graph(
        self, 
        patient_id: str, 
        timeframe: str = "1_month",
        format: str = "png",
        output_dir: str = "graphs"
    ) -> Optional[str]:
        """
        Export emotional trajectory graph for a patient.
        
        Args:
            patient_id: Patient identifier
            timeframe: Time period for the graph
            format: Output format ("png", "pdf", "svg")
            output_dir: Directory to save the graph
            
        Returns:
            Path to the exported graph file, or None if failed
        """
        try:
            # Validate access
            if not self.db_manager.validate_access(self.db_manager.user_id, patient_id, "SELECT"):
                logger.error(f"Access denied for trajectory export: user {self.db_manager.user_id}, patient {patient_id}")
                return None
            
            # Get trajectory data
            trajectory_data = self.get_patient_trajectory(patient_id, timeframe)
            
            if not trajectory_data:
                logger.warning(f"No trajectory data found for patient {patient_id}")
                return None
            
            # Create DataFrame for easier plotting
            df_data = []
            for dp in trajectory_data:
                df_data.append({
                    'timestamp': dp.timestamp,
                    'emotion': dp.emotion.value,
                    'intensity': dp.intensity,
                    'polarity': dp.polarity,
                    'session_id': dp.session_id
                })
            
            df = pd.DataFrame(df_data)
            
            # Create the visualization
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
            fig.suptitle(f'Emotional Trajectory - Patient {patient_id[-8:]} ({timeframe})', fontsize=16, fontweight='bold')
            
            # Plot 1: Polarity over time
            ax1.plot(df['timestamp'], df['polarity'], marker='o', linewidth=2, markersize=4, alpha=0.8)
            ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax1.set_ylabel('Emotional Polarity', fontsize=12)
            ax1.set_title('Emotional Polarity Over Time', fontsize=14)
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(-1.1, 1.1)
            
            # Color code by polarity
            colors = ['red' if p < -0.3 else 'orange' if p < 0 else 'lightgreen' if p < 0.3 else 'green' for p in df['polarity']]
            ax1.scatter(df['timestamp'], df['polarity'], c=colors, alpha=0.6, s=30)
            
            # Plot 2: Emotion intensity by type
            emotion_types = df['emotion'].unique()
            for emotion in emotion_types:
                emotion_data = df[df['emotion'] == emotion]
                ax2.plot(emotion_data['timestamp'], emotion_data['intensity'], 
                        marker='o', label=emotion.title(), linewidth=2, markersize=4, alpha=0.8)
            
            ax2.set_ylabel('Emotion Intensity', fontsize=12)
            ax2.set_xlabel('Time', fontsize=12)
            ax2.set_title('Emotion Intensity by Type', fontsize=14)
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1.1)
            
            # Format x-axis
            for ax in [ax1, ax2]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(trajectory_data) // 10)))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            # Add clinical annotations
            self._add_clinical_annotations(ax1, trajectory_data)
            
            plt.tight_layout()
            
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"emotion_trajectory_{patient_id[-8:]}_{timeframe}_{timestamp}.{format}"
            filepath = os.path.join(output_dir, filename)
            
            # Save the graph
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            # Audit log the export
            self.db_manager.audit_access_attempt(
                self.db_manager.user_id, patient_id, "EXPORT", "emotion_trajectory", True
            )
            
            logger.info(f"Exported emotion trajectory graph: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting graph: {e}")
            self.db_manager.audit_access_attempt(
                self.db_manager.user_id, patient_id, "EXPORT", "emotion_trajectory", False
            )
            return None

    def _persist_emotional_datapoint(self, patient_id: str, data_point: EmotionalDataPoint) -> bool:
        """Persist emotional data point to database."""
        try:
            response = self.db_manager.supabase.rpc(
                "store_emotion_trajectory_point",
                {
                    "p_patient_id": patient_id,
                    "p_timestamp": data_point.timestamp.isoformat(),
                    "p_emotion": data_point.emotion.value,
                    "p_intensity": data_point.intensity,
                    "p_polarity": data_point.polarity,
                    "p_session_id": data_point.session_id,
                    "p_chunk_id": data_point.chunk_id,
                    "p_theme": data_point.theme
                }
            ).execute()
            
            return response.data is not None
            
        except Exception as e:
            logger.error(f"Error persisting emotional data point: {e}")
            return False

    def _retrieve_trajectory_data(
        self, 
        patient_id: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[EmotionalDataPoint]:
        """Retrieve trajectory data from database."""
        try:
            response = self.db_manager.supabase.rpc(
                "get_emotion_trajectory_data",
                {
                    "p_patient_id": patient_id,
                    "p_start_date": start_date.isoformat(),
                    "p_end_date": end_date.isoformat()
                }
            ).execute()
            
            if not response.data:
                return []
            
            data_points = []
            for row in response.data:
                data_point = EmotionalDataPoint(
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    emotion=EmotionType(row["emotion"]),
                    intensity=float(row["intensity"]),
                    polarity=float(row["polarity"]),
                    session_id=row["session_id"],
                    chunk_id=row.get("chunk_id"),
                    theme=row.get("theme")
                )
                data_points.append(data_point)
            
            return data_points
            
        except Exception as e:
            logger.error(f"Error retrieving trajectory data: {e}")
            return []

    def _check_abrupt_changes(self, patient_id: str) -> None:
        """Check for abrupt changes and trigger alerts if needed."""
        try:
            alerts = self.detect_abrupt_changes(patient_id)
            
            for alert in alerts:
                if alert.severity == AlertSeverity.HIGH and alert.threshold_exceeded > 0.3:
                    self._trigger_clinical_alert(alert)
                    
        except Exception as e:
            logger.error(f"Error checking abrupt changes: {e}")

    def _log_clinical_alert(self, alert: TrajectoryAlert) -> None:
        """Log clinical alert to database."""
        try:
            self.db_manager.supabase.rpc(
                "log_clinical_alert",
                {
                    "p_patient_id": alert.patient_id,
                    "p_alert_type": alert.alert_type,
                    "p_severity": alert.severity.value,
                    "p_message": alert.message,
                    "p_threshold_exceeded": alert.threshold_exceeded,
                    "p_data_points": len(alert.data_points)
                }
            ).execute()
            
            logger.info(f"Logged clinical alert: {alert.alert_type} for patient {alert.patient_id}")
            
        except Exception as e:
            logger.error(f"Error logging clinical alert: {e}")

    def _trigger_clinical_alert(self, alert: TrajectoryAlert) -> None:
        """Trigger immediate clinical alert for high-severity issues."""
        try:
            # Log to clinical alerts table
            self.db_manager.supabase.rpc(
                "create_clinical_alert",
                {
                    "p_patient_id": alert.patient_id,
                    "p_alert_type": alert.alert_type,
                    "p_severity": "high",
                    "p_message": alert.message,
                    "p_metadata": json.dumps({
                        "threshold_exceeded": alert.threshold_exceeded,
                        "data_points_count": len(alert.data_points),
                        "detection_timestamp": alert.timestamp.isoformat()
                    })
                }
            ).execute()
            
            logger.warning(f"HIGH SEVERITY ALERT: {alert.message} for patient {alert.patient_id}")
            
        except Exception as e:
            logger.error(f"Error triggering clinical alert: {e}")

    def _add_clinical_annotations(self, ax, trajectory_data: List[EmotionalDataPoint]) -> None:
        """Add clinical annotations to the graph."""
        try:
            # Mark concerning periods
            concerning_points = [dp for dp in trajectory_data if dp.polarity < -0.5]
            if concerning_points:
                timestamps = [dp.timestamp for dp in concerning_points]
                polarities = [dp.polarity for dp in concerning_points]
                ax.scatter(timestamps, polarities, color='red', s=50, marker='x', 
                          label='Concerning Episodes', zorder=5)
            
            # Add trend line
            if len(trajectory_data) > 3:
                timestamps_numeric = mdates.date2num([dp.timestamp for dp in trajectory_data])
                polarities = [dp.polarity for dp in trajectory_data]
                
                # Calculate trend
                slope, intercept, r_value, p_value, std_err = stats.linregress(timestamps_numeric, polarities)
                
                if abs(r_value) > 0.3:  # Significant trend
                    trend_line = slope * np.array(timestamps_numeric) + intercept
                    color = 'green' if slope > 0 else 'red'
                    ax.plot([dp.timestamp for dp in trajectory_data], trend_line, 
                           color=color, linestyle='--', alpha=0.7, linewidth=2,
                           label=f'Trend (r={r_value:.2f})')
            
            if concerning_points or len(trajectory_data) > 3:
                ax.legend(loc='upper right', fontsize=10)
                
        except Exception as e:
            logger.error(f"Error adding clinical annotations: {e}")

    def get_trajectory_statistics(self, patient_id: str, timeframe: str = "1_month") -> Dict[str, Any]:
        """
        Get statistical summary of emotional trajectory.
        
        Args:
            patient_id: Patient identifier
            timeframe: Time period for analysis
            
        Returns:
            Dictionary with trajectory statistics
        """
        try:
            trajectory_data = self.get_patient_trajectory(patient_id, timeframe)
            
            if not trajectory_data:
                return {}
            
            polarities = [dp.polarity for dp in trajectory_data]
            intensities = [dp.intensity for dp in trajectory_data]
            emotions = [dp.emotion.value for dp in trajectory_data]
            
            stats_dict = {
                "total_data_points": len(trajectory_data),
                "timeframe": timeframe,
                "polarity_stats": {
                    "mean": np.mean(polarities),
                    "std": np.std(polarities),
                    "min": np.min(polarities),
                    "max": np.max(polarities),
                    "trend": self._calculate_trend(trajectory_data)
                },
                "intensity_stats": {
                    "mean": np.mean(intensities),
                    "std": np.std(intensities),
                    "min": np.min(intensities),
                    "max": np.max(intensities)
                },
                "emotion_distribution": {
                    emotion: emotions.count(emotion) / len(emotions) 
                    for emotion in set(emotions)
                },
                "concerning_episodes": len([dp for dp in trajectory_data if dp.polarity < -0.5]),
                "positive_episodes": len([dp for dp in trajectory_data if dp.polarity > 0.5])
            }
            
            return stats_dict
            
        except Exception as e:
            logger.error(f"Error calculating trajectory statistics: {e}")
            return {}

    def _calculate_trend(self, trajectory_data: List[EmotionalDataPoint]) -> Dict[str, float]:
        """Calculate trend analysis for trajectory data."""
        try:
            if len(trajectory_data) < 3:
                return {"slope": 0.0, "r_value": 0.0, "p_value": 1.0}
            
            timestamps_numeric = [(dp.timestamp - trajectory_data[0].timestamp).total_seconds() 
                                 for dp in trajectory_data]
            polarities = [dp.polarity for dp in trajectory_data]
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(timestamps_numeric, polarities)
            
            return {
                "slope": slope,
                "r_value": r_value,
                "p_value": p_value,
                "interpretation": self._interpret_trend(slope, r_value, p_value)
            }
            
        except Exception as e:
            logger.error(f"Error calculating trend: {e}")
            return {"slope": 0.0, "r_value": 0.0, "p_value": 1.0, "interpretation": "unknown"}

    def _interpret_trend(self, slope: float, r_value: float, p_value: float) -> str:
        """Interpret trend analysis results."""
        if p_value > 0.05:
            return "no_significant_trend"
        
        if abs(r_value) < 0.3:
            return "weak_trend"
        
        if slope > 0:
            return "improving" if r_value > 0.5 else "slightly_improving"
        else:
            return "declining" if r_value < -0.5 else "slightly_declining"
