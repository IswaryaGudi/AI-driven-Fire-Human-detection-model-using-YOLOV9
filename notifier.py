import smtplib
from email.mime.text import MIMEText
from twilio.rest import Client
import config


def send_email(subject: str, body: str, to_email: str | None = None):
    if not config.EMAIL_ENABLED:
        raise RuntimeError("Email is disabled")

    recipient = (to_email or config.EMAIL_RECIPIENTS).strip()
    if not recipient:
        raise RuntimeError("Recipient email missing")

    if not config.EMAIL_SENDER or not config.EMAIL_PASSWORD:
        raise RuntimeError("Email credentials missing")

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = config.EMAIL_SENDER
    msg["To"] = recipient

    with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT) as server:
        server.starttls()
        server.login(config.EMAIL_SENDER, config.EMAIL_PASSWORD)
        server.sendmail(config.EMAIL_SENDER, [recipient], msg.as_string())

    return True


def send_sms(body: str, to_phone: str | None = None):
    if not config.SMS_ENABLED:
        raise RuntimeError("SMS is disabled")

    recipient = (to_phone or config.SMS_RECIPIENTS).strip()
    if not recipient:
        raise RuntimeError("Recipient phone missing")

    if not config.TWILIO_ACCOUNT_SID or not config.TWILIO_AUTH_TOKEN or not config.TWILIO_PHONE:
        raise RuntimeError("Twilio credentials missing")

    client = Client(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)
    message = client.messages.create(
        body=body,
        from_=config.TWILIO_PHONE,
        to=recipient
    )
    return message.sid