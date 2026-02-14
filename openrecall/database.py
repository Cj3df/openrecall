import sqlite3
from collections import namedtuple
import numpy as np
from typing import Any, List, Optional, Tuple

from openrecall.config import db_path

# Define the structure of a database entry using namedtuple
Entry = namedtuple("Entry", ["id", "app", "title", "text", "timestamp", "embedding", "filename"])


def create_db() -> None:
    """
    Creates the SQLite database and the 'entries' table if they don't exist.
    Handles migrations for older database schemas.
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # Check existing columns to determine if migration is needed
            cursor.execute("PRAGMA table_info(entries)")
            columns = {info[1] for info in cursor.fetchall()}

            if columns and "filename" not in columns:
                # Migration: Add filename column and remove UNIQUE constraint from timestamp
                print("Migrating database to support multiple monitors...")
                cursor.execute(
                    """CREATE TABLE entries_new (
                           id INTEGER PRIMARY KEY AUTOINCREMENT,
                           app TEXT,
                           title TEXT,
                           text TEXT,
                           timestamp INTEGER,
                           embedding BLOB,
                           filename TEXT
                       )"""
                )
                cursor.execute(
                    """INSERT INTO entries_new (id, app, title, text, timestamp, embedding, filename)
                       SELECT id, app, title, text, timestamp, embedding, (timestamp || '.webp')
                       FROM entries"""
                )
                cursor.execute("DROP TABLE entries")
                cursor.execute("ALTER TABLE entries_new RENAME TO entries")
            elif columns:
                 # Ensure no NULL filenames exist
                 cursor.execute("UPDATE entries SET filename = (timestamp || '.webp') WHERE filename IS NULL")

            # Final table creation (for fresh installs)
            cursor.execute(
                """CREATE TABLE IF NOT EXISTS entries (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       app TEXT,
                       title TEXT,
                       text TEXT,
                       timestamp INTEGER,
                       embedding BLOB,
                       filename TEXT
                   )"""
            )
            # Add index on timestamp for faster lookups
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_timestamp ON entries (timestamp)"
            )
            conn.commit()
    except sqlite3.Error as e:
        print(f"Database error during table creation: {e}")


def get_all_entries() -> List[Entry]:
    """
    Retrieves all entries from the database.

    Returns:
        List[Entry]: A list of all entries as Entry namedtuples.
                     Returns an empty list if the table is empty or an error occurs.
    """
    entries: List[Entry] = []
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row  # Return rows as dictionary-like objects
            cursor = conn.cursor()
            cursor.execute("SELECT id, app, title, text, timestamp, embedding, filename FROM entries ORDER BY timestamp DESC")
            results = cursor.fetchall()
            for row in results:
                # Deserialize the embedding blob back into a NumPy array
                embedding = np.frombuffer(row["embedding"], dtype=np.float32) # Assuming float32, adjust if needed
                entries.append(
                    Entry(
                        id=row["id"],
                        app=row["app"],
                        title=row["title"],
                        text=row["text"],
                        timestamp=row["timestamp"],
                        embedding=embedding,
                        filename=row["filename"],
                    )
                )
    except sqlite3.Error as e:
        print(f"Database error while fetching all entries: {e}")
    return entries


def get_timestamps() -> List[Tuple[int, str]]:
    """
    Retrieves all timestamps and filenames from the database, ordered descending.

    Returns:
        List[Tuple[int, str]]: A list of all (timestamp, filename) tuples.
                   Returns an empty list if the table is empty or an error occurs.
    """
    data: List[Tuple[int, str]] = []
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            # Use the index for potentially faster retrieval
            cursor.execute("SELECT timestamp, filename FROM entries ORDER BY timestamp DESC")
            results = cursor.fetchall()
            data = [(result[0], result[1]) for result in results]
    except sqlite3.Error as e:
        print(f"Database error while fetching timestamps: {e}")
    return data


def insert_entry(
    text: str, timestamp: int, embedding: np.ndarray, app: str, title: str, filename: str
) -> Optional[int]:
    """
    Inserts a new entry into the database.

    Args:
        text (str): The extracted text content.
        timestamp (int): The Unix timestamp of the screenshot.
        embedding (np.ndarray): The embedding vector for the text.
        app (str): The name of the active application.
        title (str): The title of the active window.
        filename (str): The filename of the screenshot.

    Returns:
        Optional[int]: The ID of the newly inserted row, or None if insertion fails.
                       Prints an error message to stderr on failure.
    """
    embedding_bytes: bytes = embedding.astype(np.float32).tobytes() # Ensure consistent dtype
    last_row_id: Optional[int] = None
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO entries (text, timestamp, embedding, app, title, filename)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (text, timestamp, embedding_bytes, app, title, filename),
            )
            conn.commit()
            if cursor.rowcount > 0: # Check if insert actually happened
                last_row_id = cursor.lastrowid
            # else:
                # Optionally log that a duplicate timestamp was encountered
                # print(f"Skipped inserting entry with duplicate timestamp: {timestamp}")

    except sqlite3.Error as e:
        # More specific error handling can be added (e.g., IntegrityError for UNIQUE constraint)
        print(f"Database error during insertion: {e}")
    return last_row_id
