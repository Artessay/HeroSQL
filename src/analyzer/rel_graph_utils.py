
from torch_geometric.data import Data


class RelGraphUtils:

    @staticmethod
    def print_graph(data: Data):
        nx = data.num_nodes
        edges = list(zip(data.edge_index[0].tolist(), data.edge_index[1].tolist()))

        for idx in range(nx):
            children = [dst for src, dst in edges if src == idx]
            print(f"Node {idx} ({data.types[idx]})  --> {children}\n    Properties: {data.properties[idx]}")


if __name__ == "__main__":
    from src.analyzer.lp_parser import logical_plan_to_graph
    
    # --- 示例XML ---
    with open("case_study/demo.xml", "r") as f:
        xml_str = f.read()

    # 构建图并打印
    data = logical_plan_to_graph(xml_str)
    RelGraphUtils.print_graph(data)