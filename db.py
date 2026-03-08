import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path("app.db")

def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        event TEXT NOT NULL,
        location TEXT DEFAULT '',
        status TEXT DEFAULT 'Alert Sent'
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS settings (
        id INTEGER PRIMARY KEY CHECK (id=1),
        conf_thresh REAL NOT NULL,
        email_enabled INTEGER NOT NULL,
        sms_enabled INTEGER NOT NULL,
        email_to TEXT DEFAULT '',
        sms_to TEXT DEFAULT ''
    )
    """)

    cur.execute("SELECT id FROM settings WHERE id=1")
    row = cur.fetchone()
    if row is None:
        cur.execute("""
        INSERT INTO settings (id, conf_thresh, email_enabled, sms_enabled, email_to, sms_to)
        VALUES (1, 0.40, 1, 0, '', '')
        """)

    conn.commit()
    conn.close()

def add_log(event: str, location: str = "", status: str = "Alert Sent"):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO logs (ts, event, location, status) VALUES (?, ?, ?, ?)",
        (datetime.now().strftime("%Y-%m-%d %I:%M:%S %p"), event, location, status),
    )
    conn.commit()
    conn.close()

def list_logs(limit=50):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM logs ORDER BY id DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_settings():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM settings WHERE id=1")
    row = cur.fetchone()
    conn.close()
    return dict(row)

def update_settings(conf_thresh: float, email_enabled: int, sms_enabled: int, email_to: str, sms_to: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    UPDATE settings
    SET conf_thresh=?, email_enabled=?, sms_enabled=?, email_to=?, sms_to=?
    WHERE id=1
    """, (conf_thresh, email_enabled, sms_enabled, email_to, sms_to))
    conn.commit()
    conn.close()