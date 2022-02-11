import random
import math

def just(graph, dataset_params, N, L, alpha, beta, iters, mem):
    heterg_dictionary = dataset_params['just_heterg_dictionary']
    return original_just_walk, just_iterator(graph, N), (graph, L, heterg_dictionary, mem, alpha)

def just_iterator(graph, N):
    nodes = list(graph.nodes)
    for _ in range(N):
        random.shuffle(nodes)
        for node in nodes:
            yield (node,)

# Original implementation of JUST from https://github.com/eXascaleInfolab/JUST
# Only minimal changes performed to fit in the framework (e.g. order and name of function parameters)
def original_just_walk(start, G, path_length, heterg_dictionary, m, alpha):  # 生成一条just walks
    path = []
    path.append(start)
    homog_length = 1  # 同类点走的长度即L
    no_next_types = 0
    heterg_probability = 0
    memory_domain = []
    while len(path) < path_length:
        if no_next_types == 1:
            break
        cur = path[-1]  # 获得上一个节点 node_type,node_name
        homog_type = []
        heterg_type = []
        for node_type in heterg_dictionary:  # 同异字典 key=node_type value=hete_type
            if cur[0] == node_type:
                homog_type = node_type
                heterg_type = heterg_dictionary[node_type]
        #		print homog_type,heterg_type,cur[0],heterg_dictionary
        if not heterg_type:
            break
        if homog_type not in memory_domain:
            if len(memory_domain) < m:
                memory_domain.append(homog_type)
            else:
                memory_domain.pop(0)
                memory_domain.append(homog_type)
        else:
            memory_domain.remove(homog_type)
            memory_domain.append(homog_type)

        heterg_probability = 1 - math.pow(alpha, homog_length)  # 1-P
        r = random.uniform(0, 1)
        next_type_options = []  # 下一个节点name，不是种类
        # ——————————————————————————————————————
        if r <= heterg_probability:  # Jump

            temp = heterg_type[:]  # save heterg_typr，跳不出记忆域也要跳
            for item in memory_domain:
                if item in heterg_type:
                    heterg_type.remove(item)
            for heterg_type_iterator in heterg_type:
                next_type_options.extend([e for e in G[cur] if (e[0] == heterg_type_iterator)])
            #
            if not next_type_options:
                for heterg_type_iterator in temp:
                    next_type_options.extend([e for e in G[cur] if (e[0] == heterg_type_iterator)])
            #
            if not next_type_options:  # 没异边走，继续走同边
                next_type_options = [e for e in G[cur] if (e[0] == homog_type)]
            if not next_type_options:
                break
        # ——————————————————————————————————————
        else:  # Stay
            next_type_options = [e for e in G[cur] if (e[0] == homog_type)]
            if not next_type_options:  # 如果没同边则走异边
                for heterg_type_iterator in heterg_type:
                    next_type_options.extend([e for e in G[cur] if (e[0] == heterg_type_iterator)])
        if not next_type_options:  # 如果下一步选项为空则break
            no_next_types = 1
            break

        next_node = random.choice(next_type_options)
        path.append(next_node)
        if next_node[0] == cur[0]:
            homog_length = homog_length + 1
        else:
            homog_length = 1
    return path