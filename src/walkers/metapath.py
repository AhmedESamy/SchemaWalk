import itertools
import random
from collections import defaultdict

def metapath(graph, dataset_params, N, L, *args):
    metapaths = dataset_params['metapaths']

    neighbours = defaultdict(lambda: defaultdict(list))
    for i in graph.nodes:
        for j in graph.adj[i]:
            kind = j[0]
            neighbours[i][kind].append(j)
    neighbours = dict(neighbours)

    nodes_by_type = defaultdict(list)
    for node in graph.nodes:
        nodes_by_type[node[0]].append(node)
    nodes_by_type = dict(nodes_by_type)

    return metapath_walk_safe, metapath_walks(neighbours, metapaths, N), (neighbours, nodes_by_type, L)

def metapath_walks(neighbours, metapaths, avg_walks_per_node):
    nodes = [node for node in neighbours.keys()]
    total_walks = len(neighbours) * avg_walks_per_node
    i = 0
    while True:
        random.shuffle(nodes)
        for node in nodes:
            for metapath in metapaths:
                if metapath[0] == node[0] and metapath[1] in neighbours[node]:
                    yield (node, metapath)
                    i += 1
                    if i == total_walks:
                        return

def metapath_walk_safe(start_node, metapath, neighbours, nodes_by_type, walk_length):
    walk = [start_node]
    alternative_nodes = list(nodes_by_type[start_node[0]])
    while not recursive_walk(walk, itertools.cycle(metapath[1:]), neighbours, walk_length - 1):
        try:
            alternative_nodes.remove(walk[0])
        except:
            import sys
            print(walk, metapath, alternative_nodes[:10], file=sys.stderr, flush=True)
            raise
        walk[0] = random.choice(alternative_nodes)
    return walk

def recursive_walk(walk, kind_generator, neighbours, walk_length):
    if walk_length == 0:
        return True
    neighs = neighbours[walk[-1]][next(kind_generator)]
    random.shuffle(neighs)
    walk.append(None)
    for node in neighs:
        walk[-1] = node
        if recursive_walk(walk, kind_generator, neighbours, walk_length - 1):
            return True
    walk.pop()
    return False