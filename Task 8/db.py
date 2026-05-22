# db.py
import sqlite3
from config import DB_FILE

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tickets (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            raw_text    TEXT NOT NULL,
            customer    TEXT,
            product     TEXT,
            issue       TEXT,
            category    TEXT,
            summary     TEXT,
            status      TEXT DEFAULT 'success',
            fail_reason TEXT,
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
    print("✅ Database ready: tickets.db")

def insert_ticket(raw_text, customer, product, issue, category, summary, status="success", fail_reason=None):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO tickets (raw_text, customer, product, issue, category, summary, status, fail_reason)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (raw_text, customer, product, issue, category, summary, status, fail_reason))
    conn.commit()
    conn.close()

def insert_failed_ticket(raw_text, fail_reason):
    insert_ticket(raw_text, None, None, None, None, None, status="failed", fail_reason=fail_reason)