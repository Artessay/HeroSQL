import os

#读取的是 md 的prompt md_file_path /data1/Yangzb/Wenshu/SQL-Verifier/src/template/zero_shot_en.md
def get_system_prompt(template_file_name: str = 'zero_shot_en.md'):
    # Get the absolute path of the current Python file
    current_file_path = os.path.abspath(__file__)
    # Get the directory where the current file is located
    current_directory = os.path.dirname(current_file_path)
    # Construct the path to the zero_shot_en.md file
    md_file_path = os.path.join(current_directory, template_file_name)
    try:
        # Open and read the contents of the markdown file
        with open(md_file_path, 'r', encoding='utf-8') as file:
            system_prompt = file.read()
        return system_prompt
    except FileNotFoundError:
        print(f"Error: {template_file_name} file not found in directory {current_directory}")
        return None
    except Exception as e:
        print(f"Error: An error occurred while reading the file: {e}")
        return None

def get_query_template(template_file_name: str = 'verify_query.md'):
    return get_system_prompt(template_file_name)

def build_verify_prompt(question: str, schema: str, sql: str):
    system_prompt = get_system_prompt()
    query = get_query_template()
    query = query.format(question=question, schema=schema, sql=sql)
    return [
        {
            'role': 'system',
            'content': system_prompt
        },
        {
            'role': 'user',
            'content': query
        }
    ]

def build_generate_prompt(schema: str, question: str, evidence: str):
    system_prompt = get_system_prompt("generate_prompt.md")
    query = get_query_template("generate_query.md")
    query = query.format(schema=schema, query=question, evidence=evidence)
    return [
        {
            'role': 'system',
            'content': system_prompt
        },
        {
            'role': 'user',
            'content': query
        }
    ]



def build_verify_prompt_2(question: str, schema: str, sql: str):
    system_prompt = get_system_prompt(template_file_name="Cove_2.md")
    query = get_query_template(template_file_name="Cove_query_2.md")
    query = query.format(question=question, schema=schema, sql=sql)
    return [
        {
            'role': 'system',
            'content': system_prompt
        },
        {
            'role': 'user',
            'content': query
        }
    ]
    
def build_verify_prompt_3(question: str, schema: str, sql: str, verify_question: str):
    system_prompt = get_system_prompt(template_file_name="Cove_3.md")
    query = get_query_template(template_file_name="Cove_query_3.md")
    query = query.format(question=question, schema=schema, sql=sql,verify_question=verify_question)
    return [
        {
            'role': 'system',
            'content': system_prompt
        },
        {
            'role': 'user',
            'content': query
        }
    ]

def build_verify_prompt_4(question: str, schema: str, sql: str, verify_question: str, verify_answer: str):
    system_prompt = get_system_prompt(template_file_name="Cove_4.md")
    query = get_query_template(template_file_name="Cove_query_4.md")
    query = query.format(question=question, schema=schema, sql=sql,verify_question=verify_question,verify_answer=verify_answer)
    return [
        {
            'role': 'system',
            'content': system_prompt
        },
        {
            'role': 'user',
            'content': query
        }
    ]













# Example: Get and print the system prompt
if __name__ == "__main__":
    prompt = get_system_prompt()
    print(prompt)
    print('----------------')

    prompt = build_verify_prompt('question', 'schema', 'sql')
    print(prompt)
    print('----------------')

    prompt = build_generate_prompt('this is a schema', 'this is query', 'this is evidence')
    print(prompt)