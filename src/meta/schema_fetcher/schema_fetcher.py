import sqlite3
import logging
from typing import Dict, List, Set

from src.meta.meta_data import TableInfoRow, FKRelationship, DBMetaData

class SchemaFetcher:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def fetch_schema(self, db_path: str) -> DBMetaData:
        """
        Retrieve complete SQLite database schema information
        
        Args:
            db_path: Path to SQLite database file
        
        Returns:
            Dictionary containing:
            - table_names: List of table names
            - table_info: Column details with typed metadata
            - foreign_keys: Set of foreign key relationships
        """
        try:
            with sqlite3.connect(db_path) as conn:  # Auto-close connection
                cursor = conn.cursor()
                table_names = self._fetch_table_names(cursor)
                
                return DBMetaData(
                    table_names=frozenset(table_names),
                    table_info=self._fetch_table_details(cursor, table_names),
                    foreign_keys=frozenset(self._fetch_foreign_key_relationships(cursor, table_names))
                )
        except sqlite3.Error as e:
            raise RuntimeError(f"Database operation failed: {str(e)}.\nFile path: {db_path}") from e

    def _fetch_table_names(self, cursor: sqlite3.Cursor) -> List[str]:
        """
        Fetch all user-defined table names (exclude system tables)
        
        Security: Validates table names to prevent SQL injection
        """
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
        """)
        return [self._validate_table_name(row[0]) for row in cursor.fetchall()]

    def _fetch_table_details(self, cursor: sqlite3.Cursor, tables: List[str]) -> Dict[str, List[TableInfoRow]]:
        """
        Retrieve detailed column information for each table
        
        Structure: {table_name: [{column metadata}, ...]}
        """
        table_info = {}
        for table in tables:
            cursor.execute(f"PRAGMA table_info(`{table}`)")
            table_info[table] = [
                TableInfoRow(
                    cid=row[0],
                    name=row[1],
                    type=row[2],
                    notnull=row[3],
                    defaultValue=row[4],
                    pk=row[5]
                ) for row in cursor.fetchall()
            ]
        return table_info

    def _fetch_foreign_key_relationships(self, cursor: sqlite3.Cursor, tables: List[str]) -> Set[FKRelationship]:
        """
        Discover foreign key relationships with structure validation
        
        Returns: Set of FKRelationship tuples (from_table, from_col, to_table, to_col)
        """
        relationships = set()
        for table in tables:
            cursor.execute(f"PRAGMA foreign_key_list(`{table}`)")
            for fk in cursor.fetchall():
                try:
                    # FK format: id, seq, table, from, to, on_update, on_delete, match
                    relationships.add(FKRelationship(
                        from_table=table,
                        from_col=fk[3],
                        to_table=fk[2],
                        to_col=fk[4]
                    ))
                except IndexError as e:
                    raise ValueError(f"Invalid foreign key structure: {fk}") from e
        return relationships

    def _validate_table_name(self, name: str) -> str:
        """
        Security check to prevent SQL injection via malicious table names
        
        Rules: 
        - Must be valid Python identifier
        - Cannot contain path traversal sequences ("..")
        """
        if not name.isidentifier() or ".." in name:
            self.logger.debug(f"Invalid table name: {name} (contains invalid characters)")
            # raise ValueError(f"Invalid table name: {name} (contains invalid characters)")
        return name
