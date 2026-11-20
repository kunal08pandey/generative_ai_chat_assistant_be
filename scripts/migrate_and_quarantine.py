"""
Database Migration & Quarantine Script
========================================

Safely migrates SQLite schema to support multi-user isolation:
  1. Creates a timestamped backup of the database before making any changes.
  2. Ensures all tables (users, sessions, conversations, messages, documents) and indexes exist.
  3. Quarantines legacy records whose ownership cannot be established (sets is_quarantined = 1).
  4. Never silently assigns unowned historical records to any user.

Usage:
    python backend/scripts/migrate_and_quarantine.py
"""
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add backend to path
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from src.settings import DATABASE_PATH


def run_migration(db_path: Path = DATABASE_PATH) -> dict:
    if not db_path.exists():
        print(f"Database at {db_path} does not exist yet. Initializing fresh schema...")
        from src.db.database import init_db
        init_db()
        return {"status": "initialized_fresh", "backup": None, "quarantined_conversations": 0}

    # ── Step 1: Backup before migration ────────────────────────────────
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_path = db_path.parent / f"{db_path.stem}_backup_{timestamp}{db_path.suffix}"
    shutil.copy2(db_path, backup_path)
    print(f"[✓] Created backup at: {backup_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # ── Step 2: Ensure tables exist ────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id          TEXT PRIMARY KEY,
            email       TEXT UNIQUE NOT NULL,
            name        TEXT NOT NULL,
            picture     TEXT,
            provider    TEXT DEFAULT 'google',
            created_at  TEXT NOT NULL,
            last_login  TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            token       TEXT PRIMARY KEY,
            user_id     TEXT NOT NULL,
            created_at  TEXT NOT NULL,
            expires_at  TEXT NOT NULL,
            revoked     INTEGER NOT NULL DEFAULT 0,
            user_agent  TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id           TEXT PRIMARY KEY,
            filename     TEXT NOT NULL,
            content_type TEXT,
            chunks_count INTEGER NOT NULL DEFAULT 0,
            owner_id     TEXT,
            created_at   TEXT NOT NULL,
            FOREIGN KEY (owner_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)

    # ── Step 3: Add missing columns if upgrading existing DB ──────────
    conv_cols = {row["name"] for row in cur.execute("PRAGMA table_info(conversations)").fetchall()}
    if "owner_id" not in conv_cols:
        print("[+] Adding owner_id column to conversations")
        cur.execute("ALTER TABLE conversations ADD COLUMN owner_id TEXT")
    if "is_quarantined" not in conv_cols:
        print("[+] Adding is_quarantined column to conversations")
        cur.execute("ALTER TABLE conversations ADD COLUMN is_quarantined INTEGER NOT NULL DEFAULT 0")

    msg_cols = {row["name"] for row in cur.execute("PRAGMA table_info(messages)").fetchall()}
    if "owner_id" not in msg_cols:
        print("[+] Adding owner_id column to messages")
        cur.execute("ALTER TABLE messages ADD COLUMN owner_id TEXT")

    # ── Step 4: Quarantine unowned legacy records ──────────────────────
    # Identify conversations with no owner_id
    cur.execute("SELECT COUNT(*) AS count FROM conversations WHERE owner_id IS NULL")
    unowned_count = cur.fetchone()["count"]

    if unowned_count > 0:
        print(f"[!] Quarantining {unowned_count} legacy conversations with unverified ownership...")
        cur.execute("UPDATE conversations SET is_quarantined = 1 WHERE owner_id IS NULL")

    # Indexes
    cur.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions (user_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_conversations_owner_id ON conversations (owner_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_messages_owner_id ON messages (owner_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_owner_id ON documents (owner_id)")

    conn.commit()
    conn.close()

    print(f"[✓] Migration complete. Quarantined conversations: {unowned_count}")
    return {
        "status": "success",
        "backup": str(backup_path),
        "quarantined_conversations": unowned_count,
    }


if __name__ == "__main__":
    result = run_migration()
    print("Migration Result:", result)
