import re
import torch
from torch_geometric.data import Data

def property_to_graph(property: str):
    tokens = tokenize(property)
    ast = parse(tokens)
    return ast_to_graph(ast)


class ASTNode:
    def __init__(self, op=None, children=None, value=None):
        self.op = op
        self.children = children or []
        self.value = value

    def __repr__(self):
        if self.op:
            return f"{self.op}({', '.join(repr(c) for c in self.children)})"
        else:
            return f"{self.value}"

# def tokenize(s):
#     # 支持 '...' 和 `...`、[ ... ] 作为整体，且不保留引号/括号
#     pattern = r"""('(?:\\.|[^'])*'|`(?:\\.|[^`])*`|\[(?:\\.|[^\]])*\]|[(),]|[^\s(),'`\[\]]+)"""
#     matches = re.findall(pattern, s)
#     tokens = []
#     for token in matches:
#         if (token.startswith("'") and token.endswith("'")) \
#             or (token.startswith("`") and token.endswith("`")) \
#             or (token.startswith("[") and token.endswith("]")):
#             tokens.append(token[1:-1])
#         else:
#             tokens.append(token)
#     return tokens

def tokenize(s):
    # 支持双引号、单引号、反引号、中括号括起字符串为整体，保留其中内容（包括逗号/空格等）
    pattern = r'''("((\\.|[^"\\])*)"|'((\\.|[^'\\])*)'|`((\\.|[^`\\])*)`|\[(\\.|[^\]])*\]|[(),]|[^\s(),'"\[\]`]+)'''
    matches = re.findall(pattern, s)
    tokens = []
    for match in matches:
        token = match[0]
        # 去除两侧包裹符
        if (
            (token.startswith('"') and token.endswith('"')) or
            (token.startswith("'") and token.endswith("'")) or
            (token.startswith("`") and token.endswith("`")) or
            (token.startswith("[") and token.endswith("]"))
        ):
            tokens.append(token[1:-1])
        else:
            tokens.append(token)
    return tokens

def parse(tokens):
    def helper(idx):
        if tokens[idx] == '(':
            idx += 1
            node, idx = helper(idx)
            assert tokens[idx] == ')'
            return node, idx + 1
        elif tokens[idx] not in [',', ')']:
            op = tokens[idx]
            idx += 1
            if idx < len(tokens) and tokens[idx] == '(':
                idx += 1
                children = []
                while True:
                    if tokens[idx] == ')':
                        idx += 1
                        break
                    child, idx = helper(idx)
                    children.append(child)
                    if tokens[idx] == ',':
                        idx += 1
                return ASTNode(op, children), idx
            else:
                return ASTNode(None, value=op), idx
        else:
            raise ValueError(f'Unexpected token: {tokens[idx]}')
    node, _ = helper(0)
    return node

def ast_to_graph(ast):
    node_list = []
    edge_list = []
    def traverse(node, parent_id=None):
        node_id = len(node_list)
        node_list.append(node)
        if parent_id is not None:
            edge_list.append((node_id, parent_id))
        for child in (node.children if node.children else []):
            traverse(child, node_id)
    traverse(ast)
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty((2,0), dtype=torch.long)
    node_types = []
    node_properties = []
    for n in node_list:
        if n.op:
            node_types.append("operator")
            node_properties.append(n.op)
        else:
            node_types.append("variable")
            node_properties.append(str(n.value))
    return Data(
        num_nodes=len(node_list),
        edge_index=edge_index,
        types=node_types,
        properties=node_properties,
    )


if __name__ == '__main__':
    # s = "Amount_Settled"
    # s = "AND(>(AvgScrMath, 560), =(`Charter Funding Type`, 'Directly funded'))"
    s = "AND(=(AdmFName1, ' Dante'), =(AdmLName1, 'Alvarez'))"
    # s = "CAST(REPLACE(user_average_stars, ',', '') AS REAL)"
    tokens = tokenize(s)
    ast = parse(tokens)
    print(ast)

    data = ast_to_graph(ast)
    print(data)
    print(data.properties)

    from src.analyzer.rel_graph_utils import RelGraphUtils
    RelGraphUtils.print_graph(data)