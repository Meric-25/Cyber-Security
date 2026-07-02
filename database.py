import sqlite3
from datetime import datetime

DB_PATH = "phishing_detector.db"


def get_connection(db_path=DB_PATH):
    return sqlite3.connect(db_path)


def init_db(db_path=DB_PATH):
    conn = get_connection(db_path)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS training_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email_text TEXT NOT NULL,
            label TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS analysis_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email_snippet TEXT NOT NULL,
            prediction TEXT NOT NULL,
            phishing_probability REAL NOT NULL,
            url_count INTEGER,
            suspicious_word_count INTEGER,
            has_urgent_language INTEGER,
            analyzed_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def insert_training_row(email_text, label, db_path=DB_PATH):
    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO training_data (email_text, label, created_at) VALUES (?, ?, ?)",
        (email_text, label, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def save_analysis_result(email_text, prediction, probability, features, db_path=DB_PATH):
    conn = get_connection(db_path)
    cur = conn.cursor()
    snippet = email_text[:200]

    cur.execute("""
        INSERT INTO analysis_results
        (email_snippet, prediction, phishing_probability, url_count,
         suspicious_word_count, has_urgent_language, analyzed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        snippet,
        prediction,
        probability,
        features.get("url_count", 0),
        features.get("suspicious_word_count", 0),
        int(features.get("has_urgent_language", False)),
        datetime.now().isoformat(),
    ))

    conn.commit()
    conn.close()


def get_history(limit=20, db_path=DB_PATH):
    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT id, email_snippet, prediction, phishing_probability, analyzed_at
        FROM analysis_results
        ORDER BY analyzed_at DESC
        LIMIT ?
    """, (limit,))
    rows = cur.fetchall()
    conn.close()
    return rows


def get_statistics(db_path=DB_PATH):
    conn = get_connection(db_path)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM analysis_results")
    total = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM analysis_results WHERE prediction = 'phishing'")
    phishing_count = cur.fetchone()[0]

    conn.close()

    legit_count = total - phishing_count
    return {
        "total_analyzed": total,
        "phishing_detected": phishing_count,
        "legitimate": legit_count,
    }
