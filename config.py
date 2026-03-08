import os
from dotenv import load_dotenv

load_dotenv()

def env_bool(key: str, default: bool = False) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


# Camera
CAMERA_SOURCE = int(os.getenv("CAMERA_INDEX", "0"))

# Model
WEIGHTS_PATH = os.getenv("WEIGHTS_PATH", "best.pt")
IMG_SIZE = int(os.getenv("IMG_SIZE", "640"))
CONF_THRESHOLD = float(os.getenv("CONF_THRESH", "0.08"))
IOU_THRESHOLD = float(os.getenv("IOU_THRESH", "0.45"))

# Alerts
ALERT_COOLDOWN = int(os.getenv("ALERT_COOLDOWN", "20"))


# EMAIL CONFIGURATION
EMAIL_ENABLED = env_bool("EMAIL_ENABLED", True)

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587

EMAIL_SENDER = os.getenv("EMAIL_SENDER", "")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")
EMAIL_RECIPIENTS = os.getenv("EMAIL_RECIPIENTS", "")


# SMS CONFIGURATION (TWILIO)
SMS_ENABLED = env_bool("SMS_ENABLED", True)

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE = os.getenv("TWILIO_PHONE", "")
SMS_RECIPIENTS = os.getenv("SMS_RECIPIENTS", "")


# Server
HOST = "127.0.0.1"
PORT = 5000
DEBUG = True