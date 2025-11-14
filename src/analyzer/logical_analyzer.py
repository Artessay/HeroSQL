import logging
import os
import time
from pathlib import Path
from typing import Optional

from torch_geometric.data import Data

from src.analyzer.calcite_client import CalciteClient
from src.analyzer.rel_graph_utils import RelGraphUtils
from src.analyzer.lp_parser import logical_plan_to_graph
from src.utils.utils import find_project_root

class LogicalAnalyzer:
    """
    A class to analyze and retrieve the logical query plan for SQL statements.
    
    This class provides functionality to connect to a calcite,
    get logical query plan, and format the results for analysis.
    """
    
    def __init__(self, calcite_url: str = "http://localhost:8080/plan/"):
        """
        Initialize the query plan analyzer with a database path.
        
        Args:
            calcite_url: The URL of the calcite server
        """
        self.calcite_client = CalciteClient(calcite_url)
        self.project_path = find_project_root(Path(__file__).resolve())
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_query_plan(self, db_path: str, sql: str) -> Optional[Data]:
        """
        Complete pipeline: SQL -> logical plan -> PyG Data
        Input:
            db_path: path to the SQLite DB file.
            sql: SQL query string.
        Output:
            PyG Data instance representing the logical plan.
        """
        db_path = self.check_valid_path(db_path)
        
        # Record the start time
        start_time = time.perf_counter()
        # Get the logical query plan
        try:
            logical_plan = self.calcite_client.sql_to_logical_plan(db_path, sql, format="xml")
        except Exception as e:
            self.logger.error(f"Error getting the logical query plan: {e}\nSQL query: {sql}")
            return None
        # Record the end time
        end_time = time.perf_counter()
        # Calculate the time taken
        time_taken = end_time - start_time
        self.logger.debug(f"Time taken to get the logical query plan: {time_taken:.4f} seconds")

        # Record the start time
        start_time = time.perf_counter()
        # Convert the plan to a PyG graph
        try:
            data: Data = logical_plan_to_graph(logical_plan)
        except Exception as e:
            self.logger.error(f"Error converting the logical query plan to a PyG graph: {e}\nSQL query: {sql}")
            return None
        # Record the end time
        end_time = time.perf_counter()
        # Calculate the time taken
        time_taken = end_time - start_time
        self.logger.debug(f"Time taken to convert the logical query plan to a PyG graph: {time_taken:.4f} seconds")

        # Add SQL query to the graph
        data.sql = sql
        
        return data
    
    def check_valid_path(self, db_path: str):
        if not os.path.isabs(db_path):
            db_path = str(self.project_path.joinpath(db_path))

        if not os.path.exists(db_path):
            raise ValueError(f"Invalid database path {db_path}")
        
        return db_path

    @staticmethod 
    def print_graph(data: Data):
        RelGraphUtils.print_graph(data)

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # Initialize analyzer with database path
    analyzer = LogicalAnalyzer()
    
    # SQL query to analyze
    db_path = "/home/qrh/data/code/SQL-Verifier/data/bird/dev/dev_databases/california_schools/california_schools.sqlite"
    sql = """
        SELECT 
            frpm.CDSCode, satscores.AvgScrMath
        FROM 
            satscores INNER JOIN frpm ON satscores.cds = frpm.CDSCode
        WHERE 
            satscores.AvgScrMath > 560 AND frpm.`Charter Funding Type` = 'Directly funded'
    """
    
    # Get and print the query plan
    query_plan = analyzer.get_query_plan(db_path, sql)
    print(query_plan)
    analyzer.print_graph(query_plan)
    