import os
import time
import atexit
import cv2
from threading import Lock
from flask import Flask, Response, render_template, jsonify, request
from dotenv import load_dotenv

import config
import db
from detector import YOLOv9Detector
from notifier import send_email, send_sms

load_dotenv()

print("EMAIL_SENDER =", os.getenv("EMAIL_SENDER"))
print("EMAIL_PASSWORD =", "SET" if os.getenv("EMAIL_PASSWORD") else "MISSING")
print("EMAIL_RECIPIENTS =", os.getenv("EMAIL_RECIPIENTS"))

print("TWILIO_ACCOUNT_SID =", "SET" if os.getenv("TWILIO_ACCOUNT_SID") else "MISSING")
print("TWILIO_AUTH_TOKEN =", "SET" if os.getenv("TWILIO_AUTH_TOKEN") else "MISSING")
print("TWILIO_PHONE =", os.getenv("TWILIO_PHONE"))
print("SMS_RECIPIENTS =", os.getenv("SMS_RECIPIENTS"))

print("Using config:", config.__file__)
print("WEIGHTS_PATH:", config.WEIGHTS_PATH)

app = Flask(__name__)

# -------------------------
# Init
# -------------------------
db.init_db()

detector = YOLOv9Detector(
    fire_weights=config.WEIGHTS_PATH,
    person_weights="yolov9/yolov9-c.pt",
    fire_conf_thres=0.03,
    person_conf_thres=0.25,
    iou_thres=config.IOU_THRESHOLD,
    img_size=config.IMG_SIZE,
)

cap = cv2.VideoCapture(getattr(config, "CAMERA_SOURCE", 0))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

state_lock = Lock()
latest_counts = {"fire": 0, "person": 0}
last_alert_ts = {"fire_only": 0, "fire_human": 0}
last_snapshot_path = None


# -------------------------
# Helpers
# -------------------------
def _to_int(x):
    try:
        return int(x)
    except Exception:
        return 0


def _safe_float(x, default=0.4):
    try:
        return float(x)
    except Exception:
        return default


def _get_settings():
    try:
        s = db.get_settings()
        return s if s else {}
    except Exception:
        return {}


def normalize_detections(detections):
    out = []
    for item in detections or []:
        try:
            xyxy, label, conf = item
            x1 = int(float(xyxy[0]))
            y1 = int(float(xyxy[1]))
            x2 = int(float(xyxy[2]))
            y2 = int(float(xyxy[3]))
            out.append({
                "box": (x1, y1, x2, y2),
                "label": str(label),
                "conf": float(conf),
            })
        except Exception:
            continue
    return out


def draw_boxes(frame, detections):
    fire_count, person_count = 0, 0

    for d in detections:
        x1, y1, x2, y2 = d["box"]
        label = d["label"]
        conf = d["conf"]

        lower = label.lower()
        is_person = lower == "person"
        is_fire = "fire" in lower

        if is_person:
            person_count += 1
            color = (0, 255, 0)
        elif is_fire:
            fire_count += 1
            color = (0, 0, 255)
        else:
            color = (255, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{label} {conf:.2f}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

    return frame, fire_count, person_count


def set_runtime_confidence():
    if hasattr(detector, "set_conf"):
        try:
            detector.set_conf(0.08)
        except Exception:
            pass


def maybe_alert(alert_type: str, fire_count: int, person_count: int):
    """
    alert_type:
      - "fire_only"
      - "fire_human"
    """
    now = time.time()
    cooldown = getattr(config, "ALERT_COOLDOWN", 20)

    if now - last_alert_ts.get(alert_type, 0) < cooldown:
        return

    if alert_type == "fire_only":
        subject = "ALERT: FIRE DETECTED"
        body = (
            f"Fire detected at {time.strftime('%Y-%m-%d %I:%M:%S %p')} | "
            f"Fire Count: {fire_count} | Human Count: {person_count}"
        )
        log_event = "Fire Detected"
    elif alert_type == "fire_human":
        subject = "ALERT: FIRE + HUMAN DETECTED"
        body = (
            f"Fire and Human detected together at {time.strftime('%Y-%m-%d %I:%M:%S %p')} | "
            f"Fire Count: {fire_count} | Human Count: {person_count}"
        )
        log_event = "Fire + Human Detected"
    else:
        return

    ok_any = False
    errors = []

    try:
        if config.EMAIL_ENABLED and config.EMAIL_RECIPIENTS:
            send_email(subject, body, config.EMAIL_RECIPIENTS)
            ok_any = True
    except Exception as e:
        errors.append(f"Email: {e}")

    try:
        if config.SMS_ENABLED and config.SMS_RECIPIENTS:
            send_sms(body, config.SMS_RECIPIENTS)
            ok_any = True
    except Exception as e:
        errors.append(f"SMS: {e}")

    try:
        if ok_any:
            db.add_log(log_event, location="Camera", status="Alert Sent")
        else:
            status = "Alert Failed"
            if errors:
                status += " | " + " ; ".join(errors[:2])
            db.add_log(log_event, location="Camera", status=status)
    except Exception as e:
        print("DB log error:", e)

    if errors:
        print("Alert errors:", errors)

    last_alert_ts[alert_type] = now


def process_one_frame(frame):
    if hasattr(detector, "set_conf"):
        try:
            detector.set_conf(config.CONF_THRESHOLD)
        except Exception:
            pass

    raw = detector.detect(frame)
    detections = normalize_detections(raw)
    frame, fire_count, person_count = draw_boxes(frame, detections)

    with state_lock:
        latest_counts["fire"] = fire_count
        latest_counts["person"] = person_count

    if fire_count > 0 and person_count == 0:
        maybe_alert("fire_only", fire_count, person_count)
    elif fire_count > 0 and person_count > 0:
        maybe_alert("fire_human", fire_count, person_count)

    return frame


def read_camera_frame():
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return frame


def release_resources():
    global cap
    try:
        if cap is not None and cap.isOpened():
            cap.release()
    except Exception:
        pass


atexit.register(release_resources)


# -------------------------
# Pages
# -------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/alerts")
def alerts_page():
    return render_template("alerts.html")


@app.route("/logs")
def logs_page():
    return render_template("logs.html")


@app.route("/settings")
def settings_page():
    return render_template("settings.html")


# -------------------------
# Video endpoints
# -------------------------
def gen_frames():
    while True:
        frame = read_camera_frame()
        if frame is None:
            time.sleep(0.05)
            continue

        frame = process_one_frame(frame)

        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Cache-Control: no-cache\r\n\r\n" +
            buffer.tobytes() +
            b"\r\n"
        )


@app.route("/video")
def video():
    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.route("/frame")
def frame_image():
    frame = read_camera_frame()
    if frame is None:
        return ("", 204)

    frame = process_one_frame(frame)

    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        return ("", 204)

    return Response(
        buf.tobytes(),
        mimetype="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


# -------------------------
# APIs
# -------------------------
@app.route("/api/status")
def api_status():
    with state_lock:
        c = dict(latest_counts)

    return jsonify({
        "fires": c["fire"],
        "humans": c["person"],
        "status": "Monitoring",
    })


@app.route("/api/logs")
def api_logs():
    try:
        return jsonify(db.list_logs(limit=100))
    except Exception as e:
        print("Logs API error:", e)
        return jsonify([])


@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    if request.method == "GET":
        try:
            return jsonify(db.get_settings())
        except Exception as e:
            print("Settings GET error:", e)
            return jsonify({})

    data = request.json or {}

    conf = _safe_float(data.get("conf_thresh", 0.08), 0.08)
    email_enabled = 1 if bool(data.get("email_enabled", False)) else 0
    sms_enabled = 1 if bool(data.get("sms_enabled", False)) else 0
    email_to = str(data.get("email_to", "")).strip()
    sms_to = str(data.get("sms_to", "")).strip()

    conf = max(0.05, min(conf, 0.95))

    try:
        db.update_settings(conf, email_enabled, sms_enabled, email_to, sms_to)
        return jsonify({"ok": True})
    except Exception as e:
        print("Settings POST error:", e)
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/test_notification", methods=["POST"])
def test_notification():
    email_ok = False
    sms_ok = False
    email_error = ""
    sms_error = ""

    try:
        if config.EMAIL_ENABLED and config.EMAIL_SENDER and config.EMAIL_PASSWORD and config.EMAIL_RECIPIENTS:
            send_email("Test Alert", "Test email from fire detection project", config.EMAIL_RECIPIENTS)
            email_ok = True
        else:
            email_error = "Email config missing"
    except Exception as e:
        email_error = str(e)

    try:
        if config.SMS_ENABLED and config.TWILIO_ACCOUNT_SID and config.TWILIO_AUTH_TOKEN and config.TWILIO_PHONE and config.SMS_RECIPIENTS:
            send_sms("Test SMS from fire detection project", config.SMS_RECIPIENTS)
            sms_ok = True
        else:
            sms_error = "SMS config missing"
    except Exception as e:
        sms_error = str(e)

    return jsonify({
        "email": email_ok,
        "sms": sms_ok,
        "email_error": email_error,
        "sms_error": sms_error
    })


@app.route("/api/snapshot", methods=["POST"])
def api_snapshot():
    global last_snapshot_path

    frame = read_camera_frame()
    if frame is None:
        return jsonify({"ok": False, "error": "Camera read failed"})

    frame = process_one_frame(frame)

    ts = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs("static", exist_ok=True)
    path = os.path.join("static", f"snapshot_{ts}.jpg")

    saved = cv2.imwrite(path, frame)
    if not saved:
        return jsonify({"ok": False, "error": "Snapshot save failed"})

    last_snapshot_path = path
    return jsonify({"ok": True, "path": path})


# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    try:
        app.run(
            host=getattr(config, "HOST", "127.0.0.1"),
            port=getattr(config, "PORT", 5000),
            debug=getattr(config, "DEBUG", True),
            threaded=True,
        )
    finally:
        release_resources()