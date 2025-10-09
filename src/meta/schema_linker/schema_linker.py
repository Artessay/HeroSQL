import os
import sqlite3
from abc import abstractmethod
from typing import Dict, List

"""
This is the interface class for schema linking.
"""
class SchemaLinker():    
    @abstractmethod
    def _get_selected_columns(self, query: str, table_name: str, columns_info: List, **kwargs) -> List[str]:
        raise NotImplementedError()
