
from typing import List

from src.meta.meta_data import DBMetaData, FKRelationship, TableInfoRow


class SchemaBuilder:
    def build_schema(self, schema: DBMetaData, tight_format: bool = False):
        table_names = schema.tables

        schema_list = []
        for table_name in table_names:
            # prepare create table statement and insert statements
            if tight_format:
                create_statement = self._construct_table_description(table_name, schema)
            else:
                create_statement = self._construct_create_table_statement(table_name, schema)
            schema_list.append(create_statement)
        
        return "\n".join(schema_list) if not tight_format else " ".join(schema_list)


    def _construct_create_table_statement(self, table_name: str, schema: DBMetaData) -> str:
        columns_info: List[TableInfoRow] = schema.get_table(table_name)
        foreign_keys_map: List[FKRelationship] = schema.get_foreign_keys(table_name)
        assert columns_info is not None, f"Table {table_name} not found in schema"
        
        table_name = self._format_identifier(table_name)
        create_statement = f"CREATE TABLE {table_name} (\n"
        
        for column_info in columns_info:
            column_name = self._format_identifier(column_info["name"])
            column_type = column_info["type"]
            column_notnull = column_info["notnull"]
            column_default_value = column_info["defaultValue"]
            column_pk = column_info["pk"]
            
            # Check if the column is a primary key or foreign key
            create_statement += f"\t{column_name}\t{column_type}"
            if column_notnull:
                create_statement += "\tNOT NULL"
            else:
                create_statement += "\tNULL"
            if column_default_value is not None:
                create_statement += f"\tDEFAULT {column_default_value}"
            if column_pk:
                create_statement += "\tPRIMARY KEY"
            create_statement += ",\n"

        # Add foreign key constraints
        for fk in foreign_keys_map:
            from_col = self._format_identifier(fk.from_col)
            to_table = self._format_identifier(fk.to_table)
            to_col = self._format_identifier(fk.to_col) if fk.to_col else schema.get_primary_key(fk.to_table)
            assert to_col is not None, f"Foreign key for {from_col} not found in table {fk.to_table}"
            create_statement += f"\tforeign key ({from_col}) references {to_table} ({to_col}),\n"
        
        create_statement = create_statement.rstrip(",\n") + "\n);\n"
        return create_statement
    
    def _construct_table_description(self, table_name: str, schema: DBMetaData) -> str:
        columns_info: List[TableInfoRow] = schema.get_table(table_name)
        foreign_keys_map: List[FKRelationship] = schema.get_foreign_keys(table_name)
        assert columns_info is not None, f"Table {table_name} not found in schema"
        
        table_name = self._format_identifier(table_name)
        create_statement = f"TABLE {table_name}("
        
        for column_info in columns_info:
            column_name = self._format_identifier(column_info["name"])
            # column_type = column_info["type"]
            
            # Check if the column is a primary key or foreign key
            # create_statement += f"{column_name}:{column_type}"
            create_statement += column_name
            create_statement += ", "

        create_statement = create_statement.rstrip(", ") + ");"
        return create_statement

    @staticmethod
    def _format_identifier(identifier: str) -> str:
        """Format identifier to be enclosed in backticks if it contains spaces or special characters."""
        if " " in identifier or not identifier.isidentifier():
            return f"`{identifier}`"
        return identifier