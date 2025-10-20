"""
Detection Engine - AI/ML based intrusion detection
Uses TensorFlow for object detection and scikit-learn for anomaly detection
"""

import cv2
import numpy as np
import tensorflow as tf
from sklearn.ensemble import IsolationForest
from collections import deque
import os


class IntrusionDetector:
    """AI-powered intrusion and anomaly detector"""
    
    def __init__(self, config):
        """Initialize detector with models"""
        self.config = config
        self.model = self.load_object_detection_model()
        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        self.motion_history = deque(maxlen=30)
        self.is_trained = False
        
    def load_object_detection_model(self):
        """Load TensorFlow object detection model"""
        print("🤖 Loading TensorFlow model...")
        
        # Use MobileNetV2 for efficient detection
        model_url = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
        
        try:
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(None, None, 3), batch_size=1, dtype=tf.uint8)
            ])
            
            # Load pre-trained model from TensorFlow Hub
            import tensorflow_hub as hub
            detector = hub.load(model_url)
            
            print("✓ TensorFlow model loaded successfully")
            return detector
        except Exception as e:
            print(f"⚠ Using fallback cascade detector: {e}")
            return self.load_cascade_detector()
    
    def load_cascade_detector(self):
        """Fallback: Load Haar Cascade classifier"""
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        return cv2.CascadeClassifier(cascade_path)
    
    def analyze_frame(self, frame):
        """Analyze single frame for intrusions"""
        humans = 0
        confidence = 0.0
        
        # Detect objects/humans
        try:
            humans, confidence = self.detect_humans(frame)
        except Exception as e:
            print(f"⚠ Detection error: {e}")
            confidence = 0.0
        
        # Calculate motion-based anomaly score
        anomaly_score = self.calculate_anomaly_score(frame)
        
        return humans, confidence, anomaly_score
    
    def detect_humans(self, frame):
        """Detect humans in frame using object detection"""
        if frame is None or frame.size == 0:
            return 0, 0.0
        
        # Resize for faster processing
        height, width = frame.shape[:2]
        scale_factor = min(640 / width, 480 / height)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        resized = cv2.resize(frame, (new_width, new_height))
        
        # Try TensorFlow detection first
        try:
            humans, confidence = self.tensorflow_detect(resized)
            if humans > 0:
                return humans, confidence
        except:
            pass
        
        # Fallback to Haar Cascade (face detection)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        faces = self.model.detectMultiScale(gray, 1.1, 4)
        humans = len(faces)
        confidence = min(0.9, humans * 0.3)  # Simple confidence calculation
        
        return humans, confidence
    
    def tensorflow_detect(self, frame):
        """Detect objects using TensorFlow"""
        try:
            # Prepare input
            input_tensor = tf.convert_to_tensor(frame)
            input_tensor = tf.expand_dims(input_tensor, 0)
            input_tensor = tf.image.convert_image_dtype(input_tensor, tf.uint8)
            
            # Run detection
            results = self.model(input_tensor)
            
            # Extract human detections (class 1 is person in COCO)
            detection_classes = results['detection_classes'][0].numpy().astype(int)
            detection_scores = results['detection_scores'][0].numpy()
            
            # Count humans (persons)
            human_detections = detection_scores[detection_classes == 1]
            humans = len(human_detections[human_detections > 0.5])
            confidence = float(np.mean(human_detections)) if len(human_detections) > 0 else 0.0
            
            return humans, min(confidence, 1.0)
        except:
            return 0, 0.0
    
    def calculate_anomaly_score(self, frame):
        """Calculate motion-based anomaly score"""
        if frame is None or frame.size == 0:
            return 0.0
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # Calculate motion by differencing frames
        motion_vector = self.extract_motion_features(blurred)
        
        # Train anomaly detector on initial frames
        if len(self.motion_history) < 10:
            self.motion_history.append(motion_vector)
            if len(self.motion_history) == 10 and not self.is_trained:
                self.train_anomaly_detector()
            return 0.0
        
        # Predict anomaly
        if self.is_trained:
            prediction = self.anomaly_detector.predict([motion_vector])[0]
            confidence = self.anomaly_detector.score_samples([motion_vector])[0]
            
            # Convert to anomaly score (0-1)
            anomaly_score = max(0.0, min(1.0, -confidence / 5.0))
            
            self.motion_history.append(motion_vector)
            return anomaly_score
        
        self.motion_history.append(motion_vector)
        return 0.0
    
    def extract_motion_features(self, gray_frame):
        """Extract motion features from frame"""
        # Canny edge detection
        edges = cv2.Canny(gray_frame, 100, 200)
        
        # Calculate statistical features
        mean_intensity = np.mean(gray_frame)
        std_intensity = np.std(gray_frame)
        edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
        
        # Histogram features
        hist = cv2.calcHist([gray_frame], [0], None, [16], [0, 256])
        hist = hist.flatten() / np.sum(hist)
        
        # Combine features
        features = np.concatenate([
            [mean_intensity, std_intensity, edge_density],
            hist
        ])
        
        return features
    
    def train_anomaly_detector(self):
        """Train anomaly detector on initial normal frames"""
        if len(self.motion_history) < 10:
            return
        
        X = np.array(list(self.motion_history))
        try:
            self.anomaly_detector.fit(X)
            self.is_trained = True
            print("✓ Anomaly detector trained on baseline behavior")
        except Exception as e:
            print(f"⚠ Anomaly detector training failed: {e}")


class FrameAnnotator:
    """Utility class for annotating frames with detection results"""
    
    @staticmethod
    def draw_detections(frame, humans, confidence, anomaly_score, is_intrusion):
        """Draw detection results on frame"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # Intrusion status
        if is_intrusion:
            status_text = "⚠️  INTRUSION DETECTED"
            color = (0, 0, 255)  # Red
        else:
            status_text = "✓ Normal"
            color = (0, 255, 0)  # Green
        
        cv2.putText(frame, status_text, (10, 30), font, font_scale, color, thickness)
        
        # Statistics
        stats_text = f"Humans: {humans} | Conf: {confidence:.2f} | Anomaly: {anomaly_score:.2f}"
        cv2.putText(frame, stats_text, (10, 60), font, font_scale, (255, 255, 255), thickness)
        
        # Timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, frame.shape[0] - 10), font, font_scale, 
                   (200, 200, 200), thickness)
        
        return frame
