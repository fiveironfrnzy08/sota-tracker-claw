"""Database utilities for SOTA Tracker."""

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Union, Generator

VALID_JOURNAL_MODES = {"DELETE", "WAL", "TRUNCATE", "PERSIST", "MEMORY", "OFF"}


def configure_db_connection(db: sqlite3.Connection) -> sqlite3.Connection:
    """Apply shared SQLite connection settings."""
    journal_mode = os.environ.get("SOTA_DB_JOURNAL_MODE", "DELETE").upper()
    if journal_mode not in VALID_JOURNAL_MODES:
        journal_mode = "DELETE"
    db.execute(f"PRAGMA journal_mode={journal_mode}")
    return db


def get_db(db_path: Union[str, Path]) -> sqlite3.Connection:
    """
    Get a database connection with row factory.

    DEPRECATED: Prefer using get_db_context() for auto-closing.
    Caller is responsible for closing the connection if using this method.

    Args:
        db_path: Path to SQLite database file

    Returns:
        SQLite connection with Row factory and WAL mode enabled
    """
    db = sqlite3.connect(str(db_path), timeout=30.0)
    db.row_factory = sqlite3.Row
    return configure_db_connection(db)


@contextmanager
def get_db_context(db_path: Union[str, Path]) -> Generator[sqlite3.Connection, None, None]:
    """
    Get a database connection as a context manager that auto-closes.

    Usage:
        with get_db_context(path) as db:
            rows = db.execute(...).fetchall()
        # Connection automatically closed here

    Args:
        db_path: Path to SQLite database file

    Yields:
        SQLite connection with Row factory and WAL mode enabled
    """
    db = sqlite3.connect(str(db_path), timeout=30.0)
    db.row_factory = sqlite3.Row
    configure_db_connection(db)
    try:
        yield db
    finally:
        db.close()
