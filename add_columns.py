"""One-off SQLite migration: adds the new habit columns to productivity_logs.

Run from the project root with Flask stopped:
    python add_columns.py
Safe to run more than once (existing columns are skipped).
"""
import sqlite3
from pathlib import Path

DB = Path(__file__).resolve().parent / "instance" / "ppia.db"

NEW_COLUMNS = {
    "sleep_hours": "FLOAT",
    "disruptions": "INTEGER",
    "junk_food": "INTEGER",
    "task_difficulty": "INTEGER",
    "free_hours": "FLOAT",
    "day_type": "VARCHAR(16)",
    "weather": "VARCHAR(16)",
}


def main():
    if not DB.exists():
        raise SystemExit(f"Database not found: {DB}")
    con = sqlite3.connect(DB)
    try:
        table = None
        for (name,) in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"):
            cols = [r[1] for r in con.execute(f'PRAGMA table_info("{name}")')]
            if "study_hours" in cols:
                table, existing = name, set(cols)
                break
        if table is None:
            raise SystemExit("No table with a study_hours column found.")
        print(f"Table: {table}")
        for col, typ in NEW_COLUMNS.items():
            if col in existing:
                print(f"  skip  {col} (already there)")
            else:
                con.execute(f'ALTER TABLE "{table}" ADD COLUMN {col} {typ}')
                print(f"  added {col} {typ}")
        con.commit()
    finally:
        con.close()
    print("Done.")


if __name__ == "__main__":
    main()
