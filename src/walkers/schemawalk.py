import networkx
import random
from collections import defaultdict

def schemawalk(graph, dataset_params, N, L, alpha, *args):
    graph_adj = {}
    for i in graph.nodes:
        neighs = defaultdict(list)
        for j in graph.adj[i]:
            neighs[j[0]].append(j)
        graph_adj[i] = neighs
    num_edge_types = len(set(networkx.get_edge_attributes(graph, 'type').values()))
    return higher_order_walk, higher_order_walks(graph, N), (num_edge_types, alpha, graph_adj, L)
    
def higher_order_walks(graph, num_walks_per_node):
    nodes = list(graph.nodes())
    for _ in range(num_walks_per_node):
        random.shuffle(nodes)
        for node in nodes:
            yield (node,)

def higher_order_walk(node, num_edge_types, decay_rate, graph_adj, walk_length):
    walk = [node]
    edgeprobs = [1] * num_edge_types
    for _ in range(walk_length - 1):
        edgetypes = list(graph_adj[node].keys())
        local_edgeprobs = [edgeprobs[edgetype] for edgetype in edgetypes]
        chosen_edgetype = random.choices(edgetypes, local_edgeprobs)[0]
        node = random.choice(graph_adj[node][chosen_edgetype])
        walk.append(node)
        edgeprobs[chosen_edgetype] *= decay_rate
    return walk