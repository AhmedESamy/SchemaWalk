import networkx
import numpy as np
import random
from collections import defaultdict
from scipy.sparse import csr_matrix

def schemawalk_ho(graph, dataset_params, N, L, alpha, beta, iters, *args):
    if not hasattr(schemawalk_ho, 'graph_adj') or len(schemawalk_ho.graph_adj) != graph.number_of_nodes():
        schemawalk_ho.graph_adj = {}
        A = compute_katz(graph, beta, iters)

        nodelist = np.array(graph.nodes())
        node_types = np.unique(nodelist.view('<U1').reshape(nodelist.shape + (-1,))[:, 0]) # https://stackoverflow.com/questions/48320432/extract-the-first-letter-from-each-string-in-a-numpy-array
        node_indexes_by_type = {node_type: np.nonzero(np.char.startswith(nodelist, node_type))[0] for node_type in node_types}

        schema = defaultdict(list)
        for edge_type_id, edge_type in enumerate(np.unique(list(networkx.get_edge_attributes(graph, 'type').values()))):
            node_type_1, node_type_2 = edge_type
            node_indexes_with_type_1 = node_indexes_by_type[node_type_1]
            node_indexes_with_type_2 = node_indexes_by_type[node_type_2]
            schema[node_type_1].append((edge_type_id, nodelist[node_indexes_with_type_2], node_indexes_with_type_2))
            schema[node_type_2].append((edge_type_id, nodelist[node_indexes_with_type_1], node_indexes_with_type_1))
        schemawalk_ho.num_edge_types = edge_type_id + 1

        for node_index, node in enumerate(nodelist):
            schemawalk_ho.graph_adj[node] = {edge_type_id: (target_nodes, A[node_index, target_indexes]) for edge_type_id, target_nodes, target_indexes in schema[node[0]]}

    return schemawalk_ho_walk, schemawalk_ho_walks(graph, N), (schemawalk_ho.num_edge_types, alpha, schemawalk_ho.graph_adj, L)
    
def schemawalk_ho_walks(graph, num_walks_per_node):
    nodes = list(graph.nodes())
    for _ in range(num_walks_per_node):
        random.shuffle(nodes)
        for node in nodes:
            yield (node,)

def schemawalk_ho_walk(node, num_edge_types, decay_rate, graph_adj, walk_length):
    walk = [node]
    edgeprobs = [1] * num_edge_types
    for _ in range(walk_length - 1):
        edgetypes = list(graph_adj[node].keys())
        local_edgeprobs = [edgeprobs[edgetype] for edgetype in edgetypes]
        chosen_edgetype = random.choices(edgetypes, local_edgeprobs)[0]
        
        neighs, weights = graph_adj[node][chosen_edgetype]
        node = random.choices(neighs, weights)[0]

        walk.append(node)
        edgeprobs[chosen_edgetype] *= decay_rate
    return walk

def compute_katz(graph, beta, iters):
    A_k = networkx.to_numpy_array(graph)
    A = csr_matrix(A_k)

    beta_k = 1
    B = np.zeros_like(A_k)
    for i in range(0, iters):
        B += beta_k * A_k
        A_k = A @ A_k
        beta_k *= beta
    B += beta_k * A_k
    
    return B