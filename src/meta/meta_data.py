from dataclasses import dataclass
from typing import Dict, FrozenSet, List, NamedTuple, Optional, Set, TypedDict


#region Type Definitions (Self-documenting Data Structures)
class TableInfoRow(TypedDict):
    """Represents a single column in a table's schema"""
    cid: int          # Column ID (auto-incremented by SQLite)
    name: str         # Column name
    type: str         # Data type (e.g., INTEGER, TEXT)
    notnull: int      # 0 = nullable, 1 = not nullable
    defaultValue: str | None  # Default value, or None
    pk: int           # 0 = not primary key, 1 = primary key

FKRelationship = NamedTuple('FKRelationship', [
    ('from_table', str),  # Table containing the foreign key
    ('from_col', str),    # Column with the foreign key
    ('to_table', str),    # Referenced table
    ('to_col', str)       # Referenced column
])
#endregion

@dataclass(frozen=True)  # Immutable data container
class DBMetaData:
    """
    Strongly-typed wrapper for SQLite schema information
    Designed for easy interaction with other classes/components
    """
    table_names: FrozenSet[str]
    table_info: Dict[str, List[TableInfoRow]]
    foreign_keys: FrozenSet[FKRelationship]

    #region Convenience Properties (Read-only)
    @property
    def tables(self) -> Set[str]:
        """Alias for table_names (set interface)"""
        return set(self.table_names)

    @property
    def columns(self) -> Dict[str, List[str]]:
        """Quick access to column names by table"""
        return {
            table: [col["name"] for col in cols]
            for table, cols in self.table_info.items()
        }

    @property
    def primary_keys(self) -> Dict[str, str]:
        """Map table to primary key column"""
        return {
            table: next(
                (col["name"] for col in cols if col["pk"] == 1),
                None  # Handle tables without PK (rare in SQLite)
            )
            for table, cols in self.table_info.items()
        }
    #endregion

    #region Interaction Methods
    def get_table(self, table_name: str) -> Optional[List[TableInfoRow]]:
        """
        Get detailed column info for a specific table
        
        Returns None if table doesn't exist (instead of KeyError)
        """
        return self.table_info.get(table_name)

    def get_foreign_keys(self, table_name: str) -> List[FKRelationship]:
        """
        Get detailed foreign key info for a specific table
        
        Returns empty list if table doesn't exist (instead of KeyError)
        """
        return [
            rel for rel in self.foreign_keys
            if rel.from_table == table_name
        ]

    def get_primary_key(self, table_name: str) -> Optional[str]:
        """
        Safely retrieve primary key column for a table
        
        Raises ValueError if table doesn't exist
        """
        if table_name not in self.table_names:
            raise ValueError(f"Table not found: {table_name}")
        return self.primary_keys[table_name]

    def check_foreign_key(self, from_table: str, from_col: str) -> bool:
        """
        Check if a column is part of a foreign key relationship
        """
        return any(
            rel.from_table == from_table and rel.from_col == from_col
            for rel in self.foreign_keys
        )

    def find_referenced_table(self, from_table: str, from_col: str) -> Optional[str]:
        """
        Get the referenced table for a foreign key column
        
        Returns None if not a foreign key
        """
        for rel in self.foreign_keys:
            if rel.from_table == from_table and rel.from_col == from_col:
                return rel.to_table
        return None
    #endregion

    #region String Representation (Debugging Friendly)
    def __repr__(self) -> str:
        return (
            f"DatabaseSchema(tables={len(self.table_names)}, "
            f"foreign_keys={len(self.foreign_keys)})"
        )
    #endregion
