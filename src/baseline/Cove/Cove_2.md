You are an AI assistant tasked with generating verification questions for a given SQL query based on an original text query.  Your goal is to identify key factual elements from both the text and the SQL that need to be checked to ensure the SQL accurately reflects the user's intent from the text.

Below are a few examples demonstrating how to generate these verification questions.  Please follow this format precisely.

--- Example 1 ---
Original Text Query: "Find the customer with the highest total order amount."
Generated SQL: "SELECT C.customer_name FROM Customers C JOIN Orders O ON C.customer_id = O.customer_id GROUP BY C.customer_id ORDER BY SUM(O. total_amount) DESC LIMIT 1;"

Verification Questions:
- Is the correct entity 'customer_name' selected as requested?  Verification Question: "Does the SQL select the 'customer_name'?"
- Is the aggregation 'SUM' and column 'total_amount' correctly applied for orders?  Verification Question: "Does the SQL correctly calculate the 'SUM(O.total_amount)'?"
- Is the ordering and limit correctly applied for 'highest'?  Verification Question: "Does the SQL correctly order by 'total_amount' in 'DESCending' order and apply 'LIMIT 1'?"
- Are the necessary tables 'Customers' and 'Orders' correctly joined?  Verification Question: "Does the SQL correctly join 'Customers' and 'Orders' tables?"
- Are all conditions and clauses in the SQL directly supported by the text query?  Verification Question: "Are all clauses in the generated SQL directly derivable from the original text query?"

--- Example 2 ---
Original Text Query: "Find the number of users living in the USA and older than 30."
Generated SQL: "SELECT COUNT(*) FROM Users WHERE country = 'USA' AND age > 30;"

Verification Questions:
- Is the correct aggregate function 'COUNT(*)' used to count users?  Verification Question: "Does the SQL use 'COUNT(*)'?"
- Is the filtering condition for 'country' correctly applied?  Verification Question: "Does the SQL filter for 'country = 'USA'? '"
- Is the filtering condition for 'age' correctly applied?  Verification Question: "Does the SQL filter for 'age > 30'?"
- Are all conditions and clauses in the SQL directly supported by the text query?  Verification Question: "Are all clauses in the generated SQL directly derivable from the original text query?