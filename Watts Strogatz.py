import random
import networkx as nx
import heapq
from tqdm import tqdm
def generate_small_world_directed_graph(n, k, beta, seed=None):
    if seed is not None:
        random.seed(seed)

    # Generate undirected Watts–Strogatz graph
    G = nx.watts_strogatz_graph(n, k, beta, seed=seed)
    r = random.randint(0, n - 1)
    return G, r

def is_graph_connected(G, r):
    visited = set()
    stack = [r]

    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)

        # Visit successors of current node
        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                stack.append(neighbor)

    return len(visited) == len(G.nodes())


def dijkstra_min_depth(G, source):
    dist = {v: float('inf') for v in G.nodes}
    depth = {v: float('inf') for v in G.nodes}
    parent = {v: None for v in G.nodes}

    dist[source] = 0
    depth[source] = 0

    heap = [(0, 0, source)]  # (distance, depth, node)

    while heap:
        d_u, h_u, u = heapq.heappop(heap)

        for v in G.neighbors(u):
            w = G[u][v].get('weight', 1)  # default weight is 1 if not given
            d_v = d_u + w
            h_v = h_u + 1

            if (d_v < dist[v]) or (d_v == dist[v] and h_v < depth[v]):
                dist[v] = d_v
                depth[v] = h_v
                parent[v] = u
                heapq.heappush(heap, (d_v, h_v, v))

    return dist, depth, parent
def sample_terminals(n, q, seed=None):
    if seed is not None:
        random.seed(seed)

    return [v for v in range(n) if random.random() < q]

def count_vertices_not_on_paths(G, parent, sampled_terminals, root):
    vertices_on_paths = set()

    for t in sampled_terminals:
        current = t
        while current is not None:
            vertices_on_paths.add(current)
            current = parent.get(current)
            if current == root:
                vertices_on_paths.add(root)
                break

    all_nodes = set(G.nodes())
    not_on_paths = all_nodes - vertices_on_paths
    return len(not_on_paths)

def scheme_small_world(n, k, beta, q, times):
    removed = []
    for i in tqdm(range(times)):
        g, r = generate_small_world_directed_graph(n, k, beta)
        while not is_graph_connected(g, r):
            g, r = generate_small_world_directed_graph(n, k, beta)
        dist, depth, parent = dijkstra_min_depth(g, r)
        t = sample_terminals(n, q)
        count = count_vertices_not_on_paths(g, parent, t, r)
        removed.append(count)
    return removed

def calc_avg_and_variance(removed):
    if not removed:
        return None, None
    avg = sum(removed) / len(removed)
    variance = sum((x - avg) ** 2 for x in removed) / len(removed)
    return avg, variance

k_values = [8, 10, 14]
beta_values = [0.05, 0.2]
q_values = [0.2, 0.4]
for k in k_values:
    for beta in beta_values:
        for q in q_values:
            print(f"Running scheme with k={k}, beta={beta}, q={q}")
            removed = scheme_small_world(n=1000, k=k, beta=beta, q=q, times=100)
            avg, variance = calc_avg_and_variance(removed)
            print(f"Average vertices removed: {avg:.3f}")
            print(f"Variance of vertices removed: {variance:.3f}\n")
