"""
Video Processor - Handle video input from webcam, files, or IP streams
"""

import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from src.detector import FrameAnnotator


class VideoProcessor:
    """Process video from various sources"""
    
    def __init__(self, config):
        """Initialize video processor"""
        self.config = config
        self.cap = None
        self.frame_width = config.get('frame_width', 640)
        self.frame_height = config.get('frame_height', 480)
        self.fps = 30
        self.frame_count = 0
    
    def initialize(self, source_type, source_path=None):
        """Initialize video source"""
        try:
            if source_type == 'webcam':
                return self.init_webcam()
            elif source_type == 'video_file':
                return self.init_video_file(source_path)
            elif source_type == 'rtsp_url':
                return self.init_rtsp_stream(source_path)
            else:
                print(f"❌ Unknown source type: {source_type}")
                return False
        except Exception as e:
            print(f"❌ Initialization error: {e}")
            return False
    
    def init_webcam(self):
        """Initialize webcam"""
        print("📹 Initializing webcam...")
        
        # Try different camera indices
        for camera_index in range(5):
            self.cap = cv2.VideoCapture(camera_index)
            if self.cap.isOpened():
                print(f"✓ Webcam initialized (camera {camera_index})")
                self.setup_capture_properties()
                return True
        
        print("❌ No webcam found")
        return False
    
    def init_video_file(self, video_path):
        """Initialize video file"""
        if not video_path:
            print("❌ No video path provided")
            return False
        
        if not Path(video_path).exists():
            print(f"❌ Video file not found: {video_path}")
            return False
        
        print(f"📹 Loading video file: {video_path}")
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            print(f"❌ Failed to open video file")
            return False
        
        print(f"✓ Video file loaded")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.setup_capture_properties()
        return True
    
    def init_rtsp_stream(self, stream_url):
        """Initialize RTSP stream from IP camera"""
        if not stream_url:
            print("❌ No RTSP URL provided")
            return False
        
        print(f"📹 Connecting to RTSP stream: {stream_url}")
        
        # RTSP connection with timeout
        self.cap = cv2.VideoCapture(stream_url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer
        
        if not self.cap.isOpened():
            print(f"❌ Failed to connect to RTSP stream")
            return False
        
        print(f"✓ RTSP stream connected")
        self.setup_capture_properties()
        return True
    
    def setup_capture_properties(self):
        """Set capture properties"""
        if self.cap is None:
            return
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Reduce latency for streams
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    def get_frame(self):
        """Get next frame from source"""
        if self.cap is None or not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        
        if not ret:
            return None
        
        self.frame_count += 1
        
        # Resize frame
        frame = cv2.resize(frame, (self.frame_width, self.frame_height))
        
        return frame
    
    def display_frame(self, frame, humans, confidence, anomaly_score, is_intrusion):
        """Display frame with annotations"""
        if frame is None:
            return
        
        # Draw detections on frame
        annotated_frame = FrameAnnotator.draw_detections(
            frame.copy(), humans, confidence, anomaly_score, is_intrusion
        )
        
        # Display frame
        cv2.imshow('AI Intrusion Detection System', annotated_frame)
    
    def handle_input(self):
        """Handle keyboard input"""
        key = cv2.waitKey(1) & 0xFF
        return key
    
    def save_screenshot(self, frame):
        """Save screenshot of current frame"""
        try:
            Path('screenshots').mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshots/intrusion_{timestamp}.png"
            cv2.imwrite(filename, frame)
            print(f"📸 Screenshot saved: {filename}")
        except Exception as e:
            print(f"❌ Failed to save screenshot: {e}")
    
    def release(self):
        """Release video resources"""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print("✓ Video resources released")
