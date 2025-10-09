
from collections import Counter

def analyze_pyg_data(data_list):
    node_counts = []
    edge_counts = []
    label_counter = Counter()

    for data in data_list:
        # Number of nodes
        if hasattr(data, 'num_nodes') and data.num_nodes is not None:
            num_nodes = data.num_nodes
        else:
            num_nodes = data.x.shape[0]
        node_counts.append(num_nodes)

        # Number of edges
        num_edges = data.edge_index.size(1)
        edge_counts.append(num_edges)

        # Label distribution
        if hasattr(data, "types"):
            types = data.types
            for operator in types:
                label_counter[operator] += 1

    # Statistical output
    print("Total number of graphs:", len(data_list))
    print("Node count distribution: min={}, max={}, mean={:.2f}".format(
        min(node_counts), max(node_counts), sum(node_counts)/len(node_counts)
    ))
    print("Edge count distribution: min={}, max={}, mean={:.2f}".format(
        min(edge_counts), max(edge_counts), sum(edge_counts)/len(edge_counts)
    ))
    print("Number of label categories:", len(label_counter))
    print("Label distribution:")
    for label, count in sorted(label_counter.items()):
        print(f"  Label {label}: {count} (proportion {count/len(data_list)*100:.2f}%)")
    print("Node count distribution histogram:", Counter(node_counts))
    print("Edge count distribution histogram:", Counter(edge_counts))

if __name__ == "__main__":
    from src.utils.params import get_args
    from src.dataset.dataset_utils import load_plan_dataset
    args = get_args()
    data_list = load_plan_dataset(args.dataset, args.mode)
    analyze_pyg_data(data_list)
