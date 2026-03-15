import smtplib
from email.mime.text import MIMEText
from twilio.rest import Client
import config


def _normalize_recipients(value):
    if value is None:
        return []

    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]

    if isinstance(value, (list, tuple, set)):
        return [str(x).strip() for x in value if str(x).strip()]

    return [str(value).strip()]


def send_email(subject: str, body: str, to_email=None):
    if not config.EMAIL_ENABLED:
        raise RuntimeError("Email is disabled")

    recipients = _normalize_recipients(to_email or config.EMAIL_RECIPIENTS)

    if not recipients:
        raise RuntimeError("Recipient email missing")

    if not config.EMAIL_SENDER or not config.EMAIL_PASSWORD:
        raise RuntimeError("Email credentials missing")

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = config.EMAIL_SENDER
    msg["To"] = ", ".join(recipients)

    with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT) as server:
        server.starttls()
        server.login(config.EMAIL_SENDER, config.EMAIL_PASSWORD)
        server.sendmail(config.EMAIL_SENDER, recipients, msg.as_string())

    return True


def send_sms(body: str, to_phone=None):
    if not config.SMS_ENABLED:
        raise RuntimeError("SMS is disabled")

    recipients = _normalize_recipients(to_phone or config.SMS_RECIPIENTS)

    if not recipients:
        raise RuntimeError("Recipient phone missing")

    if not config.TWILIO_ACCOUNT_SID or not config.TWILIO_AUTH_TOKEN or not config.TWILIO_PHONE:
        raise RuntimeError("Twilio credentials missing")

    client = Client(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)

    message_sids = []
    for recipient in recipients:
        message = client.messages.create(
            body=body,
            from_=config.TWILIO_PHONE,
            to=recipient
        )
        message_sids.append(message.sid)

    return message_sids
