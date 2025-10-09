

from src.meta.schema_builder import SchemaBuilder
from src.meta.schema_fetcher import SchemaFetcher


class DBMetaAgent:
    def __init__(self):
        self.schema_fetcher = SchemaFetcher()
        self.schema_builder = SchemaBuilder()
        
    def get_schema(
            self, 
            db_path: str, 
            tight_format: bool = False
    ) -> str:
        meta_data = self.schema_fetcher.fetch_schema(db_path)
        schema_str = self.schema_builder.build_schema(meta_data, tight_format)
        return schema_str
