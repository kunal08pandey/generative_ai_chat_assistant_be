"""
Database Layer
===============

Centralised SQLite access via ``aiosqlite`` with:
  • A single configurable DB path (no more hardcoded ``"chat.db"`` scattered around).
  • WAL journal mode and foreign key enforcement.
  • Performance indexes on ``messages.conversation_id`` and ``conversations.created_at``.
  • An async dependency (``get_db``) for FastAPI route injection.
  • A synchronous ``init_db()`` for one-time table creation at startup.
"""
import sqlite3

import aiosqlite

from src.utils.logger import get_logger
from src.settings import DATABASE_PATH

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Centralised DB path — override with the ``DATABASE_URL`` env var.
# ---------------------------------------------------------------------------
DB_PATH: str = str(DATABASE_PATH)


def get_db_path() -> str:
    """Return the active database file path (useful for other modules)."""
    return DB_PATH


# ---------------------------------------------------------------------------
# Async dependency for FastAPI
# ---------------------------------------------------------------------------
async def get_db():
    """
    Yield an async ``aiosqlite`` connection for use as a FastAPI dependency.

    The connection is configured with:
      • WAL journal mode (concurrent reads).
      • Foreign key enforcement.
      • Row-factory so rows behave like dicts.
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

    Called once during application startup (before the event loop is running,
    so a synchronous ``sqlite3`` connection is fine).
    """
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Initialising database at %s", DB_PATH)
    conn = sqlite3.connect(DB_PATH, timeout=20.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row

    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id         TEXT PRIMARY KEY,
            title      TEXT NOT NULL DEFAULT 'New Chat',
            created_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id              TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            role            TEXT NOT NULL,
            content         TEXT NOT NULL,
            created_at      TEXT NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        )
    """)

    # Performance indexes
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

    conn.commit()
    conn.close()
    logger.info("Database initialised successfully — tables and indexes ready")
