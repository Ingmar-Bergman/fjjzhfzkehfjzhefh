import os
import networkx as nx
import matplotlib.pyplot as plt

def analyze_graph_folder(graph_folder):
    node_counts = []
    edge_counts = []
    edge_proportions = []

    for filename in os.listdir(graph_folder):
        if filename.endswith(".edgelist"):
            graph_path = os.path.join(graph_folder, filename)
            G = nx.read_edgelist(graph_path, data=False)  # Ignore extra data like 'id'

            num_nodes = G.number_of_nodes()
            num_edges = G.number_of_edges()
            edge_proportion = 2 * num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0

            node_counts.append(num_nodes)
            edge_counts.append(num_edges)
            edge_proportions.append(edge_proportion)

    # Plotting node count distribution
    plt.figure(figsize=(10, 5))
    plt.hist(node_counts, bins=20, color='skyblue', edgecolor='black')
    plt.title('Distribution of Number of Nodes')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Plotting edge count distribution
    plt.figure(figsize=(10, 5))
    plt.hist(edge_counts, bins=20, color='salmon', edgecolor='black')
    plt.title('Distribution of Number of Edges')
    plt.xlabel('Number of Edges')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Plotting edge proportion distribution
    plt.figure(figsize=(10, 5))
    plt.hist(edge_proportions, bins=20, color='lightgreen', edgecolor='black')
    plt.title('Distribution of Edge Proportions')
    plt.xlabel('Edge Proportion')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Path to the graph folder
graph_folder = "augmented_train/graph"  # Adjust this path as needed
analyze_graph_folder(graph_folder)
