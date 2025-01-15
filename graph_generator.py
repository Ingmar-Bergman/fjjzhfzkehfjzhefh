import os
import random
import networkx as nx
from tqdm import tqdm

# Parameters
output_folder = "synthetic_graphs_random"  # Folder to save the dataset
num_graphs = 6000  # Total number of graphs to generate
max_nodes = 50  # Maximum number of nodes per graph
min_nodes = 10   # Minimum number of nodes per graph
families = [
    "path", "cycle", "wheel", "star", "ladder", "lollipop",
    "erdos_renyi", "newman_watts_strogatz", "watts_strogatz",
    "random_regular", "barabasi_albert", "dual_barabasi_albert",
    "extended_barabasi_albert", "holme_kim", "random_lobster",
    "stochastic_block", "random_partition"
]

# Create output folders
description_folder = os.path.join(output_folder, "description")
edgelist_folder = os.path.join(output_folder, "graph")
os.makedirs(description_folder, exist_ok=True)
os.makedirs(edgelist_folder, exist_ok=True)

# Generate graphs
for i in tqdm(range(num_graphs), desc="Generating graphs"):
    # Randomly select a graph family
    family = random.choice(families)
    num_nodes = random.randint(min_nodes, max_nodes)

    # Generate graph based on the selected family
    if family == "path":
        G = nx.path_graph(num_nodes)
    elif family == "cycle":
        G = nx.cycle_graph(num_nodes)
    elif family == "wheel":
        G = nx.wheel_graph(num_nodes)
    elif family == "star":
        G = nx.star_graph(num_nodes - 1)
    elif family == "ladder":
        G = nx.ladder_graph(num_nodes // 2)
    elif family == "erdos_renyi":
        p = random.uniform(0.1, 0.3)
        G = nx.erdos_renyi_graph(num_nodes, p)
    elif family == "newman_watts_strogatz":
        k = random.randint(2, num_nodes - 1)
        p = random.uniform(0.1, 0.3)
        G = nx.newman_watts_strogatz_graph(num_nodes, k, p)
    elif family == "watts_strogatz":
        k = random.randint(2, num_nodes - 1)
        p = random.uniform(0.1, 0.3)
        G = nx.watts_strogatz_graph(num_nodes, k, p)
    elif family == "random_regular":
        while True:
            num_nodes = random.randint(min_nodes, max_nodes)
            d = random.randint(1, min(num_nodes - 1, 10))  # Ensure degree is less than num_nodes
            if (num_nodes * d) % 2 == 0:  # Check if the product is even
                break  # Exit the loop if the condition is satisfied
        G = nx.random_regular_graph(d, num_nodes)
    elif family == "barabasi_albert":
        m = random.randint(1, num_nodes // 2)
        G = nx.barabasi_albert_graph(num_nodes, m)
    elif family == "dual_barabasi_albert":
        G = nx.barabasi_albert_graph(num_nodes, random.randint(1, num_nodes // 2))
    elif family == "extended_barabasi_albert":
        G = nx.barabasi_albert_graph(num_nodes, random.randint(1, num_nodes // 2))
    elif family == "holme_kim":
        m = random.randint(1, num_nodes // 2)
        p = random.uniform(0.1, 0.3)
        G = nx.powerlaw_cluster_graph(num_nodes, m, p)
    elif family == "random_lobster":
        while True:
            # Random probabilities for the random lobster generator
            p1 = random.uniform(0.1, 0.5)  # Probability of adding to the main branch
            p2 = random.uniform(0.1, 0.5)  # Probability of adding to a secondary branch
            G = nx.random_lobster(num_nodes, p1, p2)

            # Check if the generated graph is valid
            if G.number_of_nodes() > 0 and G.number_of_edges() > 0:
                break  # Exit loop if the graph is valid
    elif family == "stochastic_block":
        # Ensure valid sizes
        max_size = max(5, num_nodes // 2)
        sizes = [random.randint(5, max_size) for _ in range(random.randint(2, 4))]
        num_communities = len(sizes)
        
        # Create a symmetric probability matrix
        p = [[0] * num_communities for _ in range(num_communities)]
        for i in range(num_communities):
            for j in range(i, num_communities):  # Ensure symmetry
                if i == j:
                    p[i][j] = random.uniform(0.3, 0.8)  # Higher probability for intra-community edges
                else:
                    prob = random.uniform(0.01, 0.1)  # Lower probability for inter-community edges
                    p[i][j] = prob
                    p[j][i] = prob  # Ensure symmetry

        G = nx.stochastic_block_model(sizes, p)

    elif family == "random_partition":
        sizes = [random.randint(5, num_nodes // 2) for _ in range(random.randint(2, 4))]
        p_in = random.uniform(0.1, 0.5)
        p_out = random.uniform(0.01, 0.1)
        G = nx.random_partition_graph(sizes, p_in, p_out)

    # Compute features
    avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
    triangles = sum(nx.triangles(G).values()) // 3
    clustering_coeff = nx.average_clustering(G)
    max_k_core = max(nx.core_number(G).values())
    num_communities = len(list(nx.algorithms.community.greedy_modularity_communities(G)))

    # Create the prompt for the graph
    prompt = (
        f"It comprises {G.number_of_nodes()} nodes and {G.number_of_edges()} edges. "
        f"The average degree is {avg_degree:.2f}, there are {triangles} triangles. "
        f"The global clustering coefficient is {clustering_coeff:.4f}, the maximum k-core is {max_k_core}. "
        f"The graph consists of {num_communities} communities."
    )

    # Save the prompt to a text file
    description_filename = os.path.join(description_folder, f"graph_{1000+i}.txt")
    with open(description_filename, "w") as txtfile:
        txtfile.write(prompt)

    # Save the edge list to an .edgelist file
    edgelist_filename = os.path.join(edgelist_folder, f"graph_{1000+i}.edgelist")
    nx.write_edgelist(G, edgelist_filename, data=True)

print(f"Generated {num_graphs} graphs. Files are saved in '{output_folder}'.")
