import os
from dotenv import load_dotenv

base_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(base_dir, ".env")

print("Looking for .env at:", env_path)
print("Exists:", os.path.exists(env_path))

load_dotenv(env_path)

print("EMAIL_SENDER:", os.getenv("EMAIL_SENDER"))
print("EMAIL_PASSWORD:", "SET" if os.getenv("EMAIL_PASSWORD") else "MISSING")
print("EMAIL_RECIPIENTS:", os.getenv("EMAIL_RECIPIENTS"))
print("TWILIO_ACCOUNT_SID:", "SET" if os.getenv("TWILIO_ACCOUNT_SID") else "MISSING")
print("TWILIO_AUTH_TOKEN:", "SET" if os.getenv("TWILIO_AUTH_TOKEN") else "MISSING")
print("TWILIO_PHONE:", os.getenv("TWILIO_PHONE"))
print("SMS_RECIPIENTS:", os.getenv("SMS_RECIPIENTS"))