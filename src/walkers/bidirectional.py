import random

def bidirectional(graph, dataset_params, N, L, *args):
    neighbours = {node: list(graph.adj[node]) for node in graph.nodes}
    return bidirectional_walk, bidirectional_iterator(neighbours, N), (neighbours, L // 2)

def bidirectional_iterator(neighbours, N):
    nodes = [node for node in neighbours.keys()]
    for _ in range(N):
        random.shuffle(nodes)
        for node in nodes:
            yield (node,)

def bidirectional_walk(start_node, neighbours, L):
    walk = [start_node]
    cur = start_node
    for _ in range(1, L):
        cur = random.choice(neighbours[cur])
        walk.append(cur)
    walk = list(reversed(walk))
    cur = start_node
    for _ in range(1, L):
        cur = random.choice(neighbours[cur])
        walk.append(cur)
    return walk