
"""
AI-Based Intrusion Detection System - Main Entry Point
Analyzes CCTV feeds to detect unauthorized access and suspicious activity
"""

import argparse
import sys
import os
import yaml
from datetime import datetime
from pathlib import Path

# Import custom modules
from src.video_processor import VideoProcessor
from src.detector import IntrusionDetector
from src.alert_handler import AlertHandler


class IntrusionDetectionSystem:
    """Main system coordinator"""
    
    def __init__(self, config_path="config.yaml"):
        """Initialize the IDS system"""
        self.config = self.load_config(config_path)
        self.video_processor = VideoProcessor(self.config)
        self.detector = IntrusionDetector(self.config)
        self.alert_handler = AlertHandler(self.config)
        self.frame_count = 0
        self.intrusion_count = 0
        
    def load_config(self, config_path):
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"✓ Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            print(f"⚠ Config file not found. Using default configuration.")
            return self.get_default_config()
    
    def get_default_config(self):
        """Return default configuration"""
        return {
            'source': 'webcam',
            'video_path': None,
            'confidence_threshold': 0.5,
            'anomaly_threshold': 0.6,
            'skip_frames': 2,
            'frame_width': 640,
            'frame_height': 480,
            'alert_type': 'console',
            'email_config': {
                'sender': '',
                'password': '',
                'recipient': ''
            }
        }
    
    def run(self, source=None, video_path=None, sensitivity=None):
        """Main execution loop"""
        print("\n" + "="*60)
        print("🔒 AI-Based Intrusion Detection System")
        print("="*60 + "\n")
        
        # Override config with command-line arguments
        if source:
            self.config['source'] = source
        if video_path:
            self.config['video_path'] = video_path
        if sensitivity:
            self.apply_sensitivity(sensitivity)
        
        # Initialize video source
        print(f"📹 Initializing video source: {self.config['source']}")
        if not self.video_processor.initialize(self.config['source'], 
                                               self.config.get('video_path')):
            print("❌ Failed to initialize video source")
            return False
        
        print("✓ Video source initialized")
        print(f"⚙ Confidence threshold: {self.config['confidence_threshold']}")
        print(f"⚙ Anomaly threshold: {self.config['anomaly_threshold']}")
        print(f"📧 Alert type: {self.config['alert_type']}\n")
        print("Press 'q' to quit, 's' to save screenshot\n")
        print("-"*60)
        
        try:
            while True:
                # Get next frame
                frame = self.video_processor.get_frame()
                if frame is None:
                    print("\n✓ End of video reached")
                    break
                
                self.frame_count += 1
                
                # Skip frames for performance
                if self.frame_count % self.config.get('skip_frames', 2) != 0:
                    continue
                
                # Detect objects and anomalies
                humans, confidence, anomaly_score = self.detector.analyze_frame(frame)
                
                # Check for intrusion
                is_intrusion = self.check_intrusion(humans, confidence, anomaly_score)
                
                # Display frame
                self.video_processor.display_frame(frame, humans, confidence, 
                                                   anomaly_score, is_intrusion)
                
                # Handle keyboard input
                key = self.video_processor.handle_input()
                if key == ord('q'):
                    print("\n✓ System shutdown by user")
                    break
                elif key == ord('s'):
                    self.video_processor.save_screenshot(frame)
                
        except KeyboardInterrupt:
            print("\n\n✓ System interrupted by user")
        finally:
            self.video_processor.release()
            self.print_summary()
    
    def check_intrusion(self, humans, confidence, anomaly_score):
        """Determine if intrusion detected"""
        is_intrusion = False
        
        # Check confidence threshold
        if confidence > self.config['confidence_threshold']:
            is_intrusion = True
        
        # Check anomaly threshold
        if anomaly_score > self.config['anomaly_threshold']:
            is_intrusion = True
        
        if is_intrusion:
            self.intrusion_count += 1
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n⚠️  INTRUSION DETECTED at {timestamp}!")
            print(f"    Humans: {humans} | Confidence: {confidence:.2f} | Anomaly: {anomaly_score:.2f}")
            
            # Send alert
            self.alert_handler.send_alert(
                timestamp=timestamp,
                humans=humans,
                confidence=confidence,
                anomaly_score=anomaly_score,
                frame_count=self.frame_count
            )
        
        return is_intrusion
    
    def apply_sensitivity(self, level):
        """Apply preset sensitivity levels"""
        if level == "high":
            self.config['confidence_threshold'] = 0.3
            self.config['anomaly_threshold'] = 0.4
        elif level == "medium":
            self.config['confidence_threshold'] = 0.5
            self.config['anomaly_threshold'] = 0.6
        elif level == "low":
            self.config['confidence_threshold'] = 0.7
            self.config['anomaly_threshold'] = 0.8
        print(f"✓ Sensitivity set to: {level}")
    
    def print_summary(self):
        """Print summary statistics"""
        print("\n" + "-"*60)
        print("📊 Session Summary")
        print("-"*60)
        print(f"Total frames processed: {self.frame_count}")
        print(f"Intrusions detected: {self.intrusion_count}")
        print(f"Detection rate: {(self.intrusion_count/max(self.frame_count, 1)*100):.2f}%")
        print("-"*60 + "\n")


def main():
    """Entry point"""
    parser = argparse.ArgumentParser(
        description='AI-Based Intrusion Detection System for CCTV'
    )
    parser.add_argument('--source', type=str, choices=['webcam', 'video_file', 'rtsp_url'],
                       help='Video source type')
    parser.add_argument('--path', type=str, help='Path to video file or RTSP URL')
    parser.add_argument('--sensitivity', type=str, choices=['low', 'medium', 'high'],
                       help='Detection sensitivity level')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    # Create logs directory if not exists
    Path('logs').mkdir(exist_ok=True)
    
    # Run system
    system = IntrusionDetectionSystem(args.config)
    system.run(source=args.source, video_path=args.path, sensitivity=args.sensitivity)


if __name__ == '__main__':
    main()
