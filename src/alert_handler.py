"""
Alert Handler - Console and Email notifications for intrusions
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import os
from pathlib import Path


class AlertHandler:
    """Handle intrusion alerts via console and email"""
    
    def __init__(self, config):
        """Initialize alert handler"""
        self.config = config
        self.alert_type = config.get('alert_type', 'console')
        self.log_file = 'logs/activity_log.txt'
        self.email_config = config.get('email_config', {})
        self.create_log_file()
    
    def create_log_file(self):
        """Create activity log file"""
        Path('logs').mkdir(exist_ok=True)
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                f.write("AI-Based Intrusion Detection System - Activity Log\n")
                f.write("="*60 + "\n\n")
    
    def send_alert(self, timestamp, humans, confidence, anomaly_score, frame_count):
        """Send intrusion alert"""
        message = self.format_alert_message(timestamp, humans, confidence, 
                                           anomaly_score, frame_count)
        
        # Console alert
        if self.alert_type in ['console', 'both']:
            self.console_alert(message)
        
        # Email alert
        if self.alert_type in ['email', 'both']:
            self.email_alert(message, timestamp, humans, confidence, anomaly_score)
        
        # Log to file
        self.log_alert(message)
    
    def format_alert_message(self, timestamp, humans, confidence, anomaly_score, frame_count):
        """Format alert message"""
        message = f"""
╔════════════════════════════════════════════════════════╗
║              🚨 INTRUSION ALERT 🚨                     ║
╚════════════════════════════════════════════════════════╝

📅 Timestamp:      {timestamp}
👥 Humans Detected: {humans}
🎯 Confidence:     {confidence:.2%}
📊 Anomaly Score:  {anomaly_score:.2f}
📹 Frame Number:   {frame_count}

⚠️  Unauthorized access or suspicious activity detected!
📍 Location:       CCTV Feed
🔔 Status:         ACTIVE THREAT

Action Required: Please investigate immediately!

════════════════════════════════════════════════════════
"""
        return message
    
    def console_alert(self, message):
        """Display alert in console"""
        print("\n" + message)
        print("🔔 Console alert sent\n")
    
    def email_alert(self, message, timestamp, humans, confidence, anomaly_score):
        """Send email alert"""
        sender_email = self.email_config.get('sender', '')
        sender_password = self.email_config.get('password', '')
        recipient_email = self.email_config.get('recipient', '')
        
        # Validate email config
        if not all([sender_email, sender_password, recipient_email]):
            print("⚠️  Email configuration incomplete. Skipping email alert.")
            print("    Please configure email settings in config.yaml")
            return False
        
        try:
            # Create email
            email_msg = MIMEMultipart('alternative')
            email_msg['Subject'] = "🚨 INTRUSION ALERT - AI Security System"
            email_msg['From'] = sender_email
            email_msg['To'] = recipient_email
            
            # Email body (HTML)
            html_body = f"""
            <html>
                <body style="font-family: Arial, sans-serif; background-color: #f5f5f5; padding: 20px;">
                    <div style="background-color: #fff; padding: 20px; border-radius: 10px; border-left: 5px solid #dc3545;">
                        <h2 style="color: #dc3545;">🚨 INTRUSION ALERT</h2>
                        <p><strong>⏰ Time:</strong> {timestamp}</p>
                        <p><strong>👥 Humans Detected:</strong> {humans}</p>
                        <p><strong>🎯 Confidence Level:</strong> {confidence:.2%}</p>
                        <p><strong>📊 Anomaly Score:</strong> {anomaly_score:.2f}</p>
                        <hr style="border: none; border-top: 1px solid #ddd;">
                        <p style="color: #666; font-size: 12px;">
                            ⚠️ Unauthorized access or suspicious activity detected on your CCTV feed.<br>
                            Please investigate immediately!
                        </p>
                    </div>
                </body>
            </html>
            """
            
            # Attach HTML content
            email_msg.attach(MIMEText(html_body, 'html'))
            
            # Send email via Gmail SMTP
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(sender_email, sender_password)
                server.send_message(email_msg)
            
            print(f"📧 Email alert sent to: {recipient_email}")
            return True
        
        except smtplib.SMTPAuthenticationError:
            print("❌ Email authentication failed!")
            print("   Check your email and app password in config.yaml")
            return False
        except smtplib.SMTPException as e:
            print(f"❌ Failed to send email: {e}")
            return False
        except Exception as e:
            print(f"❌ Email error: {e}")
            return False
    
    def log_alert(self, message):
        """Log alert to file"""
        try:
            with open(self.log_file, 'a') as f:
                f.write(message + "\n")
                f.write("-" * 60 + "\n\n")
        except Exception as e:
            print(f"⚠️  Failed to log alert: {e}")
    
    def get_log_contents(self):
        """Get activity log contents"""
        try:
            with open(self.log_file, 'r') as f:
                return f.read()
        except FileNotFoundError:
            return "No activity log found."
    
    def clear_log(self):
        """Clear activity log"""
        try:
            self.create_log_file()
            print("✓ Activity log cleared")
        except Exception as e:
            print(f"❌ Failed to clear log: {e}")
