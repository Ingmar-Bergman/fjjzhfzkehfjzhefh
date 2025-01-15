import os
import random
import tqdm

# Output file for new synthetic samples
output_file = "new_samples_12000.txt"

# Number of synthetic samples to generate
num_samples = 12000  # Adjust as needed

# Graph property ranges (replace with actual analysis results)
min_nodes = 10
max_nodes = 50
min_edges = 0
max_edges = 1200
min_avg_degree = 1.0
max_avg_degree = 20.0
min_triangles = 0
max_triangles = 1500
min_clustering_coeff = 0.0
max_clustering_coeff = 1.
min_k_core = 1
max_k_core = 50
min_num_communities = 1
max_num_communities = 50

# Template for graph descriptions
templates = [
    "This graph comprises {nodes} nodes and {edges} edges. "
    "The average degree is equal to {avg_degree} and there are {triangles} triangles in the graph. "
    "The global clustering coefficient and the graph's maximum k-core are "
    "{clustering_coeff} and {k_core} respectively. The graph consists of {num_communities} communities.",
    
    "In this graph, there are {nodes} nodes connected by {edges} edges. "
    "On average, each node is connected to {avg_degree} other nodes. Within the graph, "
    "there are {triangles} triangles, forming closed loops of nodes. "
    "The global clustering coefficient is {clustering_coeff}. Additionally, the graph has a maximum "
    "k-core of {k_core} and a number of communities equal to {num_communities}.",
]

# Generate and save new graph descriptions
with open(output_file, 'w') as f:
    for i in tqdm.tqdm(range(num_samples), desc="Generating new samples"):
        # Randomly sample properties
        nodes = random.randint(min_nodes, max_nodes)
        edges = random.randint(min_edges, min(nodes * (nodes - 1) // 2, max_edges))
        avg_degree = random.uniform(min_avg_degree, min(max_avg_degree, 2 * edges / nodes))
        triangles = random.randint(min_triangles, min(max_triangles, edges // 3))
        clustering_coeff = random.uniform(min_clustering_coeff, max_clustering_coeff)
        k_core = random.randint(min_k_core, max_k_core)
        num_communities = random.randint(min_num_communities, max_num_communities)
        
        # Use a random template for variety
        template = random.choice(templates)
        description = template.format(
            nodes=nodes, edges=edges, avg_degree=avg_degree,
            triangles=triangles, clustering_coeff=clustering_coeff,
            k_core=k_core, num_communities=num_communities
        )
        
        graph_name = f"new_graph_{i}"
        line = f"{graph_name},{description}\n"
        f.write(line)

print(f"New samples generated and saved to '{output_file}'")
