import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
from typing import List, Dict, Any
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertSystem:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.email_enabled = self.config.get("email_alerts", {}).get("enabled", False)
        self.email_recipients = self.config.get("email_alerts", {}).get("recipients", [])
        self.email_sender = self.config.get("email_alerts", {}).get("sender", "")
        self.email_server = self.config.get("email_alerts", {}).get("server", "")
        self.email_port = self.config.get("email_alerts", {}).get("port", 587)
        self.email_username = self.config.get("email_alerts", {}).get("username", "")
        self.email_password = self.config.get("email_alerts", {}).get("password", "")

        self.sms_enabled = self.config.get("sms_alerts", {}).get("enabled", False)
        self.sms_recipients = self.config.get("sms_alerts", {}).get("recipients", [])
        self.sms_api_url = self.config.get("sms_alerts", {}).get("api_url", "")
        self.sms_api_key = self.config.get("sms_alerts", {}).get("api_key", "")

    def send_email_alert(self, subject: str, message: str) -> None:
        """Send an email alert to the configured recipients."""
        if not self.email_enabled:
            logger.info("Email alerts are disabled.")
            return

        msg = MIMEMultipart()
        msg['From'] = self.email_sender
        msg['To'] = ", ".join(self.email_recipients)
        msg['Subject'] = subject

        msg.attach(MIMEText(message, 'plain'))

        try:
            server = smtplib.SMTP(self.email_server, self.email_port)
            server.starttls()
            server.login(self.email_username, self.email_password)
            text = msg.as_string()
            server.sendmail(self.email_sender, self.email_recipients, text)
            server.quit()
            logger.info(f"Email alert sent successfully to {self.email_recipients}")
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")

    def send_sms_alert(self, message: str) -> None:
        """Send an SMS alert to the configured recipients."""
        if not self.sms_enabled:
            logger.info("SMS alerts are disabled.")
            return

        headers = {
            "Authorization": f"Bearer {self.sms_api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "recipients": self.sms_recipients,
            "message": message
        }

        try:
            response = requests.post(self.sms_api_url, json=data, headers=headers)
            response.raise_for_status()
            logger.info(f"SMS alert sent successfully to {self.sms_recipients}")
        except Exception as e:
            logger.error(f"Failed to send SMS alert: {e}")

    def trigger_alert(self, event_type: str, details: Dict[str, Any]) -> None:
        """Trigger an alert based on the type of event."""
        subject = f"Market Making HFT Alert: {event_type}"
        message = f"Event Type: {event_type}\nDetails:\n"
        for key, value in details.items():
            message += f"{key}: {value}\n"

        # Send email and SMS alerts
        self.send_email_alert(subject, message)
        self.send_sms_alert(message)

    def critical_error_alert(self, error_message: str) -> None:
        """Send a critical error alert."""
        logger.error(f"Critical error occurred: {error_message}")
        self.trigger_alert("Critical Error", {"error_message": error_message})

    def order_execution_alert(self, order_id: str, status: str, details: Dict[str, Any]) -> None:
        """Send an alert for order execution."""
        logger.info(f"Order execution alert triggered for order {order_id} with status {status}.")
        self.trigger_alert("Order Execution", {"order_id": order_id, "status": status, **details})

    def risk_management_alert(self, risk_event: str, details: Dict[str, Any]) -> None:
        """Send an alert for risk management events."""
        logger.warning(f"Risk management alert: {risk_event}")
        self.trigger_alert("Risk Management", {"risk_event": risk_event, **details})

    def manual_override_alert(self, override_event: str, details: Dict[str, Any]) -> None:
        """Send an alert for manual override events."""
        logger.info(f"Manual override alert: {override_event}")
        self.trigger_alert("Manual Override", {"override_event": override_event, **details})

    def performance_alert(self, metrics: Dict[str, float]) -> None:
        """Send an alert related to performance metrics."""
        logger.info("Performance metrics alert triggered.")
        self.trigger_alert("Performance Metrics", metrics)

    def market_condition_alert(self, condition: str, details: Dict[str, Any]) -> None:
        """Send an alert for significant market condition changes."""
        logger.info(f"Market condition alert: {condition}")
        self.trigger_alert("Market Condition", {"condition": condition, **details})
