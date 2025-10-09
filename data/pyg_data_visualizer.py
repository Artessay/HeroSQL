import matplotlib.pyplot as plt

def plot_graph_size_distribution(data_list, save_path=None):
    """
    Plots a histogram showing the distribution of the number of nodes per graph in the dataset.
    Args:
        data_list (List[torch_geometric.data.Data]): List of PyG Data objects.
        save_path (str, optional): If provided, saves the figure to this path.
    """
    # Collect node count for each graph
    node_counts = []
    for data in data_list:
        num_nodes = data.num_nodes
        node_counts.append(num_nodes)
        # if num_nodes == 1:
        #     print(data.types)
        #     print(data.properties)

    # Plot configuration
    plt.figure(figsize=(6, 4))
    max_num, min_num = max(node_counts), min(node_counts)
    print(min_num, max_num)
    bins = range(min_num, max_num + 2) # To cover 2-15 (last bin edge exclusive)
    # plt.hist(node_counts, bins=bins, color='#4C72B0', edgecolor='black', alpha=0.85)
    plt.hist(node_counts, bins=bins, color='#4C72B0', edgecolor='black', alpha=0.85, density=True)
    
    # Calculate mean node count
    mean_nodes = sum(node_counts) / len(node_counts)

    # Draw mean line
    plt.axvline(mean_nodes, color='#dd8452', linestyle='--', linewidth=2, label=f'Mean = {mean_nodes:.2f}')

    # Axis & title
    plt.xlabel("Number of nodes per graph", fontsize=12)
    # plt.ylabel("Number of graphs", fontsize=12)
    plt.ylabel("Proportion of graphs", fontsize=12)
    # plt.title("Node Count Distribution in Dataset", fontsize=14, pad=14)
    plt.legend(fontsize=11)
    plt.xticks(range(min_num, max_num + 1))
    plt.tight_layout()

    # Optional save
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    from src.utils.params import get_args
    from src.dataset.dataset_utils import load_plan_dataset
    args = get_args()
    dataset, mode = args.dataset, args.mode
    data_list = load_plan_dataset(dataset, mode)
    plot_graph_size_distribution(data_list, save_path=f'data/figures/node_distribution_{dataset}_{mode}.png')