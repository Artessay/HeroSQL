import importlib

from .schema_linker import SchemaLinker


def load_schema_linker(schema_linking_strategy: str) -> SchemaLinker:
    schema_linking_module = importlib.import_module('src.schema_linking.' + schema_linking_strategy)
    schema_linker = getattr(schema_linking_module, schema_linking_strategy)()
    return schema_linker