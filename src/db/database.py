"""
Database Layer
===============

Centralised SQLite access via ``aiosqlite`` with:
  • A single configurable DB path.
  • WAL journal mode and foreign key enforcement.
  • Performance indexes on ``messages.conversation_id`` and ``conversations.created_at``.
  • Session management table for secure HttpOnly-cookie authentication.
  • Owner-scoped conversations, messages, documents, and artifacts.
  • Quarantine tables for legacy unassigned records.
  • An async dependency (``get_db``) for FastAPI route injection.
  • A synchronous ``init_db()`` for one-time table creation at startup.
"""
import sqlite3
from pathlib import Path

import aiosqlite

from src.utils.logger import get_logger
from src.settings import DATABASE_PATH

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Centralised DB path — override with the ``DATABASE_URL`` env var.
# ---------------------------------------------------------------------------
DB_PATH: str = str(DATABASE_PATH)


def get_db_path() -> str:
    """Return the active database file path."""
    return DB_PATH


# ---------------------------------------------------------------------------
# Async dependency for FastAPI
# ---------------------------------------------------------------------------
async def get_db():
    """
    Yield an async ``aiosqlite`` connection for use as a FastAPI dependency.
    """
    db = await aiosqlite.connect(DB_PATH, timeout=20.0)
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA foreign_keys=ON")
    try:
        yield db
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# Startup initialisation (synchronous, called once)
# ---------------------------------------------------------------------------
def init_db() -> None:
    """
    Create tables and indexes if they don't exist.
    """
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Initialising database at %s", DB_PATH)
    conn = sqlite3.connect(DB_PATH, timeout=20.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row

    cur = conn.cursor()

    # ── Users ──────────────────────────────────────────────────────────
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

    # ── Sessions (Phase 1: secure HttpOnly cookie auth) ────────────────
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

    # ── Conversations (with owner_id for data isolation) ───────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id          TEXT PRIMARY KEY,
            title       TEXT NOT NULL DEFAULT 'New Chat',
            created_at  TEXT NOT NULL,
            owner_id    TEXT,
            is_quarantined INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY (owner_id) REFERENCES users(id) ON DELETE SET NULL
        )
    """)

    # ── Messages (with owner_id for data isolation) ────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id              TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            role            TEXT NOT NULL,
            content         TEXT NOT NULL,
            created_at      TEXT NOT NULL,
            owner_id        TEXT,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
            FOREIGN KEY (owner_id) REFERENCES users(id) ON DELETE SET NULL
        )
    """)

    # ── Documents (Phase 1 & 3: owner-scoped document management) ─────
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

    # ── Performance indexes ────────────────────────────────────────────
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_messages_conversation_id
        ON messages (conversation_id)
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_conversations_created_at
        ON conversations (created_at DESC)
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_messages_created_at
        ON messages (created_at ASC)
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_users_email
        ON users (email)
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_sessions_user_id
        ON sessions (user_id)
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_sessions_expires_at
        ON sessions (expires_at)
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_conversations_owner_id
        ON conversations (owner_id)
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_messages_owner_id
        ON messages (owner_id)
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_documents_owner_id
        ON documents (owner_id)
    """)

    conn.commit()
    conn.close()
    logger.info("Database initialised successfully — tables and indexes ready")
