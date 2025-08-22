"""
validate_schema.py

This script validates the consistency of two SQL schema files:

    1. schema_raw.sql   -> exported directly from PostgreSQL (contains the full dump)
    2. schema.sql       -> the "clean" schema you maintain in your repo for users

The goal is to ensure that your maintained schema.sql is an accurate reflection
of the raw database schema.

How it works:
-------------
1. Both files are read and parsed for `CREATE TABLE ... (...)` blocks.
2. For each table:
   - Extracts the table name.
   - Extracts column names and their declared types.
   - Ignores constraints (PRIMARY KEY, FOREIGN KEY, etc.) to focus on columns.
3. Compares the results:
   - Detects missing tables between RAW and CLEAN schemas.
   - Detects missing columns within matching tables.
   - Reports differences in column data types.

Output:
-------
- Prints a summary of the number of tables parsed from each schema.
- Reports:
  * ❌ Tables or columns missing from one schema.
  * ⚠️ Column type mismatches between RAW and CLEAN.

Usage:
------
    python validate_schema.py

This will print a human-readable report of differences so you can
verify whether `schema.sql` is correct and up-to-date.

Next steps:
-----------
You can extend this script to:
- Automatically generate a "fix" schema file by copying missing
  definitions from `schema_raw.sql`.
- Normalize all column types (e.g., force everything to TEXT if
  preparing for DuckDB migration).

"""

import re
from pathlib import Path

RAW_FILE = Path("schema_raw.sql")
CLEAN_FILE = Path("schema.sql")


def extract_tables(sql_text: str):
    """
    Extract CREATE TABLE blocks and return dict:
    {
        table_name: {column_name: column_type, ...}
    }
    """
    tables = {}
    create_table_regex = re.compile(
        r"CREATE TABLE\s+(\w+)\s*\((.*?)\);",
        re.DOTALL | re.IGNORECASE,
    )

    for match in create_table_regex.finditer(sql_text):
        table = match.group(1).lower()
        body = match.group(2)

        columns = {}
        for line in body.splitlines():
            line = line.strip().strip(",")
            if not line or line.upper().startswith(("CONSTRAINT", "PRIMARY", "FOREIGN", "UNIQUE", "CHECK")):
                continue
            parts = line.split()
            col = parts[0].lower()
            ctype = parts[1].lower() if len(parts) > 1 else "unknown"
            columns[col] = ctype

        tables[table] = columns

    return tables


def compare_tables(raw_tables, clean_tables):
    """
    Compare extracted schemas and print differences.
    """
    raw_only = set(raw_tables.keys()) - set(clean_tables.keys())
    clean_only = set(clean_tables.keys()) - set(raw_tables.keys())

    if raw_only:
        print("❌ Tables missing in CLEAN schema:", raw_only)
    if clean_only:
        print("❌ Tables missing in RAW schema:", clean_only)

    for table in sorted(set(raw_tables.keys()) & set(clean_tables.keys())):
        raw_cols = raw_tables[table]
        clean_cols = clean_tables[table]

        raw_only_cols = set(raw_cols) - set(clean_cols)
        clean_only_cols = set(clean_cols) - set(raw_cols)

        if raw_only_cols:
            print(f"❌ {table}: columns missing in CLEAN: {raw_only_cols}")
        if clean_only_cols:
            print(f"❌ {table}: columns missing in RAW: {clean_only_cols}")

        # Compare column types
        for col in set(raw_cols) & set(clean_cols):
            raw_type = raw_cols[col]
            clean_type = clean_cols[col]
            if raw_type != clean_type:
                print(f"⚠️ {table}.{col}: RAW={raw_type}, CLEAN={clean_type}")


def main():
    raw_sql = RAW_FILE.read_text(encoding="utf-8")
    clean_sql = CLEAN_FILE.read_text(encoding="utf-8")

    raw_tables = extract_tables(raw_sql)
    clean_tables = extract_tables(clean_sql)

    print(f"Parsed {len(raw_tables)} tables from RAW and {len(clean_tables)} from CLEAN.")

    compare_tables(raw_tables, clean_tables)


if __name__ == "__main__":
    main()
