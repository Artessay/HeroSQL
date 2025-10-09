try:
    from .llm_generator import LLMGenerator
except ImportError:
    class LLMGenerator:
        pass

class SqlGenerator(LLMGenerator):
    def __init__(self, model_path: str, num_samples: int = 10):
        super().__init__(model_path, num_samples)

    def __call__(self, prompts: list):
        responses = self.generate(prompts)
        return self.postprocess(responses)

    @staticmethod
    def postprocess(responses: str):
        answers = []
        for response in responses:
            sqls = [SqlGenerator.fetch_code(sql, code_type="sql", default=";") for sql in response]
            # clean_sqls = [SqlGenerator.remove_sql_comments(sql) for sql in sqls]
            # formatted_sqls = [sql.replace('\t',' ').replace('\n',' ') for sql in clean_sqls]
            answer = [sql for sql in sqls if sql != ";"]
            answer = list(set(answer))
            answers.append(answer)

        return answers

    @staticmethod
    def fetch_code(response: str, code_type: str, default: str = "") -> str:
        # fetch code block
        if "```" in response:
            code = response.split("```")[1]
        else:
            code = response

        # remove code type from the beginning of the code block
        length = len(code_type) + 1
        if f'{code_type}\n' in code[:length]:
            code = code[length:]

        # if code is empty, return default value
        if code == "":
            code = default

        return code
    
    
    @staticmethod
    def remove_sql_comments(sql):
        """
        Remove all SQL comments (both inline and full-line) from a SQL string.
        
        Args:
            sql (str): Input SQL string potentially containing comments.
            
        Returns:
            str: SQL string with all comments removed while preserving string literals.
            
        Handles:
        - Both single (') and double (") quoted strings
        - Escaped quotes within strings (e.g. 'It''s valid')
        - Comments appearing anywhere in the SQL
        - Preservation of newlines (to be replaced later)
        """
        result = []
        in_string = None  # Track string state: None, 'single', or 'double'
        i = 0
        n = len(sql)
        
        while i < n:
            char = sql[i]
            
            if in_string:
                # Handle quoted string content
                if (in_string == 'single' and char == "'") or (in_string == 'double' and char == '"'):
                    # Check for escaped quote (SQL-standard '' or "")
                    if i + 1 < n and sql[i+1] == char:
                        result.extend([char, char])
                        i += 2  # Skip escaped quote
                    else:
                        in_string = None
                        result.append(char)
                        i += 1
                else:
                    result.append(char)
                    i += 1
            else:
                # Handle comment detection
                if char == '-' and i + 1 < n and sql[i+1] == '-':
                    # Skip all characters until end of line
                    while i < n and sql[i] not in ('\n', '\r'):
                        i += 1
                elif char in ("'", '"'):
                    in_string = 'single' if char == "'" else 'double'
                    result.append(char)
                    i += 1
                else:
                    result.append(char)
                    i += 1
                    
        return ''.join(result)

if __name__ == '__main__':
    responses = [
        "```sql\nSELECT * FROM users WHERE age > 18;\n```",
        "SQL Code: \n```sql\nSELECT * -- how are you?\n FROM users -- name\n WHERE age > 18;\n```",
    ]

    print(SqlGenerator.postprocess(responses))