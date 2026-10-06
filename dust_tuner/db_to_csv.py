import csv
import os
import sqlite3

DB_PATH = "submissions.db"
CSV_PATH = "submissions.csv"
TABLE = "submissions"

conn = sqlite3.connect(DB_PATH)

try:
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {TABLE} ORDER BY id")
    columns = [description[0] for description in cursor.description]

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in cursor:
            writer.writerow(row)

finally:
    conn.close()