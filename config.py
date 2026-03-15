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
CONF_THRESHOLD = float(os.getenv("CONF_THRESH", "0.30"))
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

# -----------------------------
# ALERT GROUPS
# -----------------------------
ALERT_GROUPS = {
    "admin_team": {
        "emails": [
            "ishugudi31@gmail.com",
            "iswaryagudi793@gmail.com",
        ],
        "phones": [
            "+917981199533",
            "+916302365106",
        ],
    },
    "safety_team": {
        "emails": [
            "2022csm.r44@svce.edu.in",
        ],
        "phones": [
            "+919849256291",
        ],
    },
}


def get_group_emails(group_name: str):
    return ALERT_GROUPS.get(group_name, {}).get("emails", [])


def get_group_phones(group_name: str):
    return ALERT_GROUPS.get(group_name, {}).get("phones", [])


# Server
HOST = "127.0.0.1"
PORT = 5000
DEBUG = True
