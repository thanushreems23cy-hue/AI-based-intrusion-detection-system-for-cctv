"""
Utility functions for AI Intrusion Detection System
Located in: src/utils.py
"""

import os
import json
from pathlib import Path
from datetime import datetime, timedelta


class DataManager:
    """Manage system data and logs"""
    
    @staticmethod
    def ensure_directories():
        """Create necessary directories"""
        directories = ['data', 'logs', 'screenshots', 'models']
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
    
    @staticmethod
    def get_system_stats():
        """Get system statistics"""
        try:
            with open('logs/activity_log.txt', 'r') as f:
                content = f.read()
                intrusions = content.count('INTRUSION DETECTED')
            return {
                'intrusions': intrusions,
                'log_file_size': os.path.getsize('logs/activity_log.txt')
            }
        except:
            return {'intrusions': 0, 'log_file_size': 0}
    
    @staticmethod
    def cleanup_old_logs(days=7):
        """Remove logs older than specified days"""
        cutoff = datetime.now() - timedelta(days=days)
        try:
            for screenshot in Path('screenshots').glob('*.png'):
                if datetime.fromtimestamp(screenshot.stat().st_mtime) < cutoff:
                    screenshot.unlink()
                    print(f"Deleted old screenshot: {screenshot}")
        except Exception as e:
            print(f"Cleanup error: {e}")


class PerformanceMonitor:
    """Monitor system performance"""
    
    def __init__(self):
        self.fps_history = []
        self.processing_times = []
    
    def calculate_fps(self, time_elapsed):
        """Calculate frames per second"""
        if time_elapsed > 0:
            fps = 1.0 / time_elapsed
            self.fps_history.append(fps)
            return fps
        return 0
    
    def get_average_fps(self):
        """Get average FPS"""
        if self.fps_history:
            return sum(self.fps_history) / len(self.fps_history)
        return 0
    
    def get_system_info(self):
        """Get system information"""
        import platform
        return {
            'system': platform.system(),
            'processor': platform.processor(),
            'python_version': platform.python_version()
        }


class ConfigValidator:
    """Validate configuration file"""
    
    @staticmethod
    def validate(config):
        """Validate configuration"""
        errors = []
        
        # Check source
        if config.get('source') not in ['webcam', 'video_file', 'rtsp_url']:
            errors.append("Invalid source type")
        
        # Check thresholds
        if not (0 <= config.get('confidence_threshold', 0.5) <= 1):
            errors.append("Confidence threshold must be between 0 and 1")
        
        if not (0 <= config.get('anomaly_threshold', 0.6) <= 1):
            errors.append("Anomaly threshold must be between 0 and 1")
        
        # Check alert type
        if config.get('alert_type') not in ['console', 'email', 'both']:
            errors.append("Invalid alert type")
        
        return len(errors) == 0, errors


# src/__init__.py
"""
AI Intrusion Detection System Package
"""

__version__ = "1.0.0"
__author__ = "Security Team"
__description__ = "AI-powered intrusion detection for CCTV feeds"

from src.detector import IntrusionDetector
from src.alert_handler import AlertHandler
from src.video_processor import VideoProcessor

__all__ = [
    'IntrusionDetector',
    'AlertHandler',
    'VideoProcessor'
]
