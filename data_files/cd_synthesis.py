import os
import numpy as np
import torch
import networkx as nx
from typing import Tuple


def generate_random_dag(n_nodes: int, edge_prob=0.3, seed=None) -> Tuple[nx.DiGraph, np.ndarray]:
    """
    Generate a random DAG
    """
    if seed is not None:
        np.random.seed(seed)

    adj_matrix = np.triu((np.random.rand(n_nodes, n_nodes) < edge_prob).astype(int), k=1)

    perm = np.random.permutation(n_nodes)
    adj_matrix = adj_matrix[perm][:, perm]

    G = nx.DiGraph(adj_matrix)
    return G, adj_matrix


def simulate_linear_gaussian(G, n_samples=1000, noise_scale=1.0, seed=None):
    """
    use Linear Gaussian SEM to generate observation
    """
    if seed is not None:
        np.random.seed(seed)

    n_nodes = G.number_of_nodes()
    data = np.zeros((n_samples, n_nodes))

    order = list(nx.topological_sort(G))

    W = np.zeros((n_nodes, n_nodes))
    for i, j in G.edges():
        W[i, j] = np.random.uniform(0.5, 2.0) * np.random.choice([-1, 1])

    for node in order:
        parents = list(G.predecessors(node))
        if parents:
            data[:, node] = data[:, parents] @ W[parents, node] + noise_scale * np.random.randn(n_samples)
        else:
            data[:, node] = noise_scale * np.random.randn(n_samples)

    return data, W


def generate_split_datasets(
    n_graphs=30,
    min_nodes=5,
    max_nodes=10,
    n_samples=200,
    edge_prob=0.3,
    out_dir="synthetic_dataset",
    seed=0
):
    """
    Generate multiple DAGs and observations, split it into train/val/test with 6/2/2
    """
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    # train/val/test
    n_train = int(0.6 * n_graphs)
    n_val = int(0.2 * n_graphs)
    n_test = n_graphs - n_train - n_val
    splits = {
        "train": n_train,
        "val": n_val,
        "test": n_test
    }

    for split_name, split_num in splits.items():
        data_dict = {
            "src_list": [],
            "dst_list": [],
            "x_n_list": [],
            "y_list": []
        }

        for g_id in range(split_num):
            n_nodes = np.random.randint(min_nodes, max_nodes + 1)
            G, adj = generate_random_dag(n_nodes, edge_prob=edge_prob, seed=seed + g_id)

            data, W = simulate_linear_gaussian(G, n_samples=n_samples, noise_scale=1.0, seed=seed + g_id)

            # edge list
            if G.number_of_edges() > 0:
                src, dst = zip(*G.edges())
            else:
                src, dst = ([], [])
            src = torch.tensor(src, dtype=torch.long)
            dst = torch.tensor(dst, dtype=torch.long)

            # set node feature with node id
            x_n = torch.arange(n_nodes, dtype=torch.long)

            # y = observation (n_samples × n_nodes)
            y = torch.tensor(data, dtype=torch.float32)

            data_dict["src_list"].append(src)
            data_dict["dst_list"].append(dst)
            data_dict["x_n_list"].append(x_n)
            data_dict["y_list"].append(y)

        save_path = os.path.join(out_dir, f"{split_name}.pth")
        torch.save(data_dict, save_path)
        print(f"✅ Saved {split_name} set with {split_num} DAGs to {save_path}")


if __name__ == "__main__":
    generate_split_datasets(
        n_graphs=1000,
        min_nodes=5,
        max_nodes=20,
        n_samples=100,
        out_dir="./cd_syn_processed/",
        seed=42
    )
