import random
import networkx as nx
import heapq
from tqdm import tqdm

def load_random_connected_component(path, directed=True, min_size=50):
    if directed:
        G = nx.read_edgelist(path, create_using=nx.DiGraph(), nodetype=int)
        components = list(nx.weakly_connected_components(G))
    else:
        G = nx.read_edgelist(path, create_using=nx.Graph(), nodetype=int)
        components = list(nx.connected_components(G))

    # Filter components by minimum size to avoid trivial ones
    components = [c for c in components if len(c) >= min_size]
    chosen = random.choice(components)

    G_sub = G.subgraph(chosen).copy()
    G_sub = nx.convert_node_labels_to_integers(G_sub)
    r = random.choice(list(G_sub.nodes()))
    return G_sub, r


def is_graph_connected(G, r):
    visited = set()
    stack = [r]

    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)

        # Visit successors of current node
        for neighbor in G.successors(node):
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

        for v in G.successors(u):
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

def scheme_random_components(path, q, times, directed=True, min_size=50):
    removed = []
    for _ in tqdm(range(times)):
        g, r = load_random_connected_component(path, directed, min_size)
        dist, depth, parent = dijkstra_min_depth(g, r)
        t = sample_terminals(len(g), q)
        ratio = count_vertices_not_on_paths(g, parent, t, r) / len(g)
        removed.append(ratio)
    return removed



def calc_avg_and_variance(removed):
    if not removed:
        return None, None
    avg = sum(removed) / len(removed)
    variance = sum((x - avg) ** 2 for x in removed) / len(removed)
    return avg, variance

qs = [0.2, 0.4, 0.6]
for q in qs:
    removed = scheme_random_components("email-Enron.txt", q=q, times=100, directed=True)
    avg, var = calc_avg_and_variance(removed)
    print(f"Real graph (email-Enron) with q={q}: Avg removed = {avg:.2f}, Var = {var:.6f}")
