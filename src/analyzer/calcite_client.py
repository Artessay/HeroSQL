import logging
import os
from typing import Optional

import requests


class CalciteClient:
    """
    A class to analyze and retrieve the logical query plan for SQL statements.
    
    This class provides functionality to connect to a calcite,
    get logical query plan, and format the results for analysis.
    """
    
    def __init__(self, calcite_url: str = "http://localhost:8080/plan/"):
        """
        Initialize the query plan analyzer with a database path.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.calcite_url = calcite_url


    def sql_to_logical_plan(self, db_path: str, sql: str, format: str = "text") -> Optional[str]:
        """
        Get the logical query plan for a given SQL statement.
        
        Args:
            sql: The SQL statement to analyze
            
        Returns:
            A dictionary containing the query plan if successful, None otherwise
        """
        assert format in ["text", "xml", "json"]

        payload = {
            "dbPath": db_path,
            "sql": sql
        }
        calcite_url = os.path.join(self.calcite_url, format)
        logging.debug(f"Sending payload {payload} to {calcite_url}")
        
        response = requests.post(calcite_url, json=payload)
        if response.status_code != 200:
            raise RuntimeError(f"Calcite HTTP error {response.status_code} for {response.text} with payload {payload}")
        
        logical_plan: str = response.text
        return logical_plan.strip()

# Example usage
if __name__ == "__main__":
    client = CalciteClient("http://localhost:8080/plan/")
    
    # SQL query to analyze
    db_path = "/home/qrh/data/code/SQL-Verifier/data/bird/dev/dev_databases/california_schools/california_schools.sqlite"
    
    # question: Among the schools with the average score in Math over 560 in the SAT test, how many schools are directly charter-funded?
    sql = """
        SELECT 
            COUNT(T2.`School Code`) 
        FROM 
            satscores AS T1 INNER JOIN frpm AS T2 
            ON T1.cds = T2.CDSCode 
        WHERE 
            T1.AvgScrMath > 560 AND T2.`Charter Funding Type` = 'Directly funded'
    """
    
    # db_path = "/home/qrh/data/code/SQL-Verifier/data/spider/train/train_databases/insurance_policies/insurance_policies.sqlite"
    # sql = "SELECT Amount_Settled, Amount_Claimed FROM Claims ORDER BY Amount_Settled ASC LIMIT 1"
    
    # db_path = "/home/qrh/data/code/SQL-Verifier/data/spider/train/train_databases/icfp_1/icfp_1.sqlite"
    # sql = "SELECT T1.lname FROM Authors AS T1 JOIN Authorship AS T2 ON T1.authID = T2.authID JOIN Papers AS T3 ON T2.paperID = T3.paperID WHERE T3.title = 'Binders Unbound'"

    # db_path = "/home/qrh/data/code/SQL-Verifier/data/bird/dev/dev_databases/california_schools/california_schools.sqlite"
    # sql = "SELECT T1.Website FROM schools AS T1 INNER JOIN ( SELECT CDSCode FROM schools WHERE AdmFName1 = 'Mike' AND AdmLName1 = 'Larson' UNION SELECT CDSCode FROM schools WHERE AdmFName1 = ' Dante' AND AdmLName1 = 'Alvarez' ) AS T2 ON T1.CDSCode = T2.CDSCode"

    # db_path = "/home/qrh/data/bird/dev/dev_databases/debit_card_specializing/debit_card_specializing.sqlite"
    # sql = "SELECT p.Description FROM products p JOIN transactions_1k t ON p.ProductID = t.ProductID GROUP BY p.Description ORDER BY SUM(t.Amount) DESC LIMIT 10"

    db_path = "/home/qrh/data/code/SQL-Verifier/data/bird/dev/dev_databases/student_club/student_club.sqlite"
    sql = "SELECT zip_code FROM zip_code WHERE state = 'PR' AND type = 'PO Box'"

    # Get and print the query plan
    logical_plan = client.sql_to_logical_plan(db_path, sql, format="text")
    print(logical_plan)
    