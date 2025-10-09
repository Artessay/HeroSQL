import xml.etree.ElementTree as ET

import torch
from torch_geometric.data import Data

from src.analyzer.ast_parser import property_to_graph

def logical_plan_to_graph(xml_str):
    """
    Convert the logical query plan to a PyTorch Geometric graph.
    
    Args:
        xml_str: The logical query plan XML string
        
    Returns:
        A PyTorch Geometric Data object representing the graph
    """
    root = ET.fromstring(xml_str)
    
    return xml_to_graph(root)


def xml_to_graph(root: ET.Element):
    nodes, edges = [], []
    _traverse_relnodes(root, nodes, edges)
    
    return _build_graph(nodes, edges)

def _traverse_relnodes(root, nodes, edges, next_id = 0, parent_idx = None):
    """
    Post-order traversal to assign ids to nodes.
    Each node gets its id after all its children are processed.
    Args:
        root: XML node to process
        nodes: list to store nodes information
        edges: list to store edges (child_id, parent_id)
        next_id: current available id (int)
        parent_idx: id of parent node (int or None)
    Returns:
        next_id after all children and this node are processed.
        Returns current node id for parent edge connection.
    """
    if root is None:
        return next_id, None

    child_ids = []

    # Recursively traverse children first (post-order)
    inputs = root.find('./Inputs')
    if inputs is not None:
        for input_node in inputs.findall('./RelNode'):
            next_id, child_id = _traverse_relnodes(input_node, nodes, edges, next_id)
            child_ids.append(child_id)

    # Assign current node ID
    node_id = next_id
    next_id += 1

    rel_type = root.attrib.get('type', 'Unknown')
    rel_type = rel_type.replace('Jdbc', '') # remove Jdbc prefix
    rel_type = rel_type.replace('Logical', '') # remove Logical prefix
    prop_str = _parse_properties(root)
    condition_graph = _parse_condition(root)

    nodes.append({
        'id': node_id,
        'type': rel_type,
        'property': prop_str,
        'condition': condition_graph
    })

    # Add edges from children to current node
    for child_id in child_ids:
        edges.append((child_id, node_id))

    # If there is a parent, add edge from current node to parent
    if parent_idx is not None:
        edges.append((node_id, parent_idx))

    return next_id, node_id  # return updated next_id and current node id
    

def _parse_properties(node):
    props = []
    for prop in node.findall('./Property'):
        name = prop.attrib['name']
        value = prop.text.strip() if prop.text else ""
        props.append(f"{name}=[{value}]")
    return ', '.join(props)

def _parse_condition(node):
    for prop in node.findall('./Property'):
        name = prop.attrib['name']
        value = prop.text.strip() if prop.text else ""

        if name == 'condition':
            return property_to_graph(value)
    
    return Data(
        num_nodes=0,
        edge_index=[],
    )


def _build_graph(nodes, edges):
    node_types = [n['type'] for n in nodes]
    node_properties = [n['property'] for n in nodes]
    node_conditions = [n['condition'] for n in nodes]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2,0), dtype=torch.long)
    
    return Data(
        num_nodes=len(nodes),
        edge_index=edge_index,
        types=node_types,
        properties=node_properties,
        conditions=node_conditions,
    )
