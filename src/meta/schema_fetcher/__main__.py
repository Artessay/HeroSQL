"""
SQLite Schema Fetcher - Command Line Tool

Usage: python -m schema_fetcher /path/to/database.sqlite

Example:
python -m src.meta.schema_fetcher data/bird/dev/dev_databases/financial/financial.sqlite
"""
if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path
    from pprint import pprint

    from src.meta.schema_fetcher import SchemaFetcher

    parser = argparse.ArgumentParser(
        description="Retrieve and display SQLite database schema",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("db_path", help="Path to SQLite database file")
    args = parser.parse_args()

    # Validate database existence
    db_file = Path(args.db_path)
    if not db_file.is_file():
        print(f"Error: Database file not found - {args.db_path}", file=sys.stderr)
        sys.exit(1)

    try:
        # Fetch and pretty-print schema
        schema = SchemaFetcher().fetch_schema(args.db_path)
        pprint(schema, indent=2, sort_dicts=False)
        
        # Optional: Print human-readable summary
        print(f"\nDiscovered {len(schema['table_names'])} tables with "
              f"{len(schema['foreign_keys'])} foreign key relationships")
        
    except Exception as e:
        print(f"Schema fetch failed: {str(e)}", file=sys.stderr)
        sys.exit(1)
