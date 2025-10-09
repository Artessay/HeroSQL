import csv
import hashlib
import json
import logging
import random

from sqlglot import exp, parse_one
from tqdm import tqdm

OPPOSITE_OPS = {
    exp.GT: exp.LTE,
    exp.LT: exp.GTE,
    exp.EQ: exp.NEQ,
    exp.GTE: exp.LT,
    exp.LTE: exp.GT,
    exp.NEQ: exp.EQ,
}

def csv_to_list(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        return [row[0] for row in reader]
    
string_list= csv_to_list('preprocess/vocabulary.csv')

CONSTANT_RULES = {
    "str_mapping": string_list,
    "number_strategy": {
        "range": (1, 32767),         # 默认随机范围
        "column_based_ranges": {     # 根据列名动态调整范围
            "age": (18, 80),
            "salary": (3000, 20000),
        }
    },
}

COLUMN_CANDIDATES = []

def reverse_comparison(node,reverse_prob=0.7):
    opposite_class = OPPOSITE_OPS.get(node.__class__)
    if opposite_class:
        if random.random() < reverse_prob:
            return opposite_class(**node.args)
    return node
def replace_logic_operator(node,reverse_prob=0.7):
    if random.random() < reverse_prob:
        if isinstance(node, exp.And):
            return exp.Or(this=node.this, expression=node.expression)
        elif isinstance(node, exp.Or):
            return exp.And(this=node.this, expression=node.expression)
    return node
def replace_aggregate(node):
    if isinstance(node, (exp.Avg, exp.Max, exp.Min)):
        # print(1)
        new_func = random.choice([exp.Avg, exp.Max, exp.Min])
        # print(new_func)
        return new_func(**node.args)
    return node
GROUP_BY_CANDIDATES = ['score', 'department', 'class_id', 'exam_num', 'region']
def modify_group_by(node):
    if isinstance(node, exp.Group) and len(COLUMN_CANDIDATES) > 0:
        selected = random.sample(COLUMN_CANDIDATES, k=1) # 随机选择1-3个字段
        
        # 生成AST表达式节点列表
        expressions = [parse_one(field) for field in selected]
        node.set('expressions', expressions)
    return node
def modify_constants(node, context=None, replace_prob=0.8):
    if isinstance(node, exp.Literal):
        if random.random() > replace_prob:
            return node 
       
        column_name = _get_parent_column(node) if context else None
     
   
        if node.is_string:
            return _replace_string(node, column_name)
        elif node.is_number:
            return _replace_number(node, column_name)
        
    return node
def _get_parent_column(node):
    parent = node.parent
    while parent:
        if isinstance(parent, exp.Column):
            return parent.name
        parent = parent.parent
    return None
def _replace_string(node, column_name):
 
    value = node.this.strip("'\"")  
  
    candidates = CONSTANT_RULES["str_mapping"]
    new_value = random.choice(candidates)
    return exp.Literal.string(new_value) 
def _replace_number(node, column_name):
    default_min, default_max = CONSTANT_RULES["number_strategy"]["range"]
    
    if column_name:
        col_ranges = CONSTANT_RULES["number_strategy"]["column_based_ranges"]
        min_val, max_val = col_ranges.get(column_name, (default_min, default_max))
    else:
        min_val, max_val = default_min, default_max
    
    new_value = random.randint(min_val, max_val)
    return exp.Literal.number(new_value)  
def apply_random_rules(ast):
    rules = [reverse_comparison, replace_logic_operator, replace_aggregate, modify_constants] # modify_group_by
    selected_rules = random.sample(rules, k=random.randint(1, 3))  # 每次随机选择1-3个规则
    for rule in selected_rules:
        ast = ast.transform(rule)
    return ast
def normalize_sql(sql):
    # 统一关键字大写、移除多余空格、标准化引号
    ast = parse_one(sql)
    return ast.sql(pretty=True, normalize=True)  # 自动标准化语法
def sql_fingerprint(sql):
    normalized = normalize_sql(sql)
    return hashlib.md5(normalized.encode()).hexdigest()

def augment_negative_sql(sql):
    try:
        ast = parse_one(sql, dialect="sqlite")
    except Exception as e:
        logging.debug(f"SQL parse failed: {str(e)}.")
        return None
    
    max_retries = 5
    negtive_sql = None
    while max_retries > 0:
        try:
            ast = apply_random_rules(ast)  
            negtive_sql = ast.sql()
            break
        except Exception as e:
            logging.debug(f"SQl transform failed: {str(e)}")
            max_retries -= 1
    
    return negtive_sql

def data_augmentation(original_sql, num_variants=5):
    variants = []
    fingerprints = set()
    max_attempts = num_variants * 10  # 防止无限循环
    try :
        while len(variants) < num_variants and max_attempts > 0:
            ast = parse_one(original_sql,dialect="sqlite")
            ast = apply_random_rules(ast)  
            variant_sql = ast.sql()
            fingerprint = sql_fingerprint(variant_sql)
            
            if fingerprint not in fingerprints:
                fingerprints.add(fingerprint)
                variants.append(variant_sql)
            max_attempts -= 1
    except Exception as e:
        logging.debug(f"SQL parse failed: {str(e)}.")
        
    return variants[:num_variants]  # 返回指定数量的非重复结果
def get_results(data):
    enhanced_data = []
    for item in tqdm(data):
        new_item = item.copy()
        new_item["enhanceSQL"] = data_augmentation(item["SQL"])
        enhanced_data.append(new_item)
    return enhanced_data

if __name__ == "__main__":

    with open('raw/bird-train.json') as f:
        data = json.load(f)
    result=get_results(data)
    with open('bird/train.json', 'w') as f:
        json.dump(result, f, indent=4)
   