__author__ = 'Alex'
import networkx as nx
from operator import itemgetter
from itertools import groupby
import numpy as np

def labels_for_cut_height(n, clustergraph, cut_height):
    """ cuts a given clustergraph at cut_height and returns list of clustersets
    :param clustergraph: c
    :param cut_height:
    :param startnodes:
    :return:
    """

    label_gen = labels_for_cut_height_generator(n, clustergraph)
    report_labels = [-1]*n
    for (height, labels) in label_gen:
        if height > cut_height:
            break
        else:
            report_labels = labels
    return report_labels


def labels_for_cut_height_generator(n, clustergraph, report_heights=False):
    """ cuts a given clustergraph at cut_height and returns list of clustersets
    :param clustergraph: c
    :param cut_height:
    :param startnodes:
    :return:
    """
    edge_triples = [((x, y), clustergraph.node[y]["height"]) for (x,y) in clustergraph.edges()]
    edge_triples.extend([((-1, x), clustergraph.node[x]["height"]) for (x, y) in clustergraph.in_degree().items() if y == 0])
    edge_triples = [list(g) for k, g in groupby(sorted(edge_triples, key=itemgetter(1), reverse=True), lambda x :x[1])]
    edge_triples = [(k[0], list(i)) for i, k in [zip(*i) for i in edge_triples]]
    edge_triples = sorted(edge_triples, key=itemgetter(0))
    clusters = set()

    if report_heights:
        yield -1, [-1]*n, [-1]
    else:
        yield -1, [-1]*n
    for height, edges in edge_triples:
        old_clusters = set()
        for (source, target) in edges:
            #new_clusters.append(frozenset(clustergraph.node[target]["nodes"]))
            clusters.add(target)
            for neighbor in clustergraph.neighbors(target):
                old_clusters.add(neighbor)

            #if source != -1:
                #old_clusters.add(frozenset(clustergraph.node[source]["nodes"]))
                #old_clusters.add(source)

        for old_cluster in old_clusters:
            clusters.discard(old_cluster)

        #clusters.update(new_clusters)
        clusterlist = [clustergraph.node[x]["nodes"] for x in clusters]
        item_occurrences = [[y for y in range(len(clusterlist)) if x in clusterlist[y]] for x in range(n)]

        if report_heights:
            # clusterheights = [clustergraph.node[x]["height"] for x in clusters]
            clusterheights = {x: clustergraph.node[list(clusters)[x]]["height"] for x in range(len(clusters))}
            yield height, [i[0] if len(i) == 1 else -1 if len(i) == 0 else -2 for i in item_occurrences], clusterheights
        else:
            yield height, [i[0] if len(i) == 1 else -1 if len(i) == 0 else -2 for i in item_occurrences]

    return


def height_dif_generator(n, clustergraph):
    """ cuts a given clustergraph at cut_height and returns list of clustersets
    :param clustergraph: c
    :param cut_height:
    :param startnodes:
    :return:
    """
    edge_triples = [((x, y), clustergraph.node[y]["height"]) for (x,y) in clustergraph.edges()]
    edge_triples.extend([((-1, x), clustergraph.node[x]["height"]) for (x, y) in clustergraph.in_degree().items() if y == 0])
    edge_triples = [list(g) for k, g in groupby(sorted(edge_triples, key=itemgetter(1), reverse=True), lambda x :x[1])]
    edge_triples = [(k[0], list(i)) for i, k in [zip(*i) for i in edge_triples]]
    edge_triples = sorted(edge_triples, key=itemgetter(0))
    clusters = set()

    for height, edges in edge_triples:
        old_clusters = set()
        for (source, target) in edges:
            #new_clusters.append(frozenset(clustergraph.node[target]["nodes"]))
            clusters.add(target)
            for neighbor in clustergraph.neighbors(target):
                old_clusters.add(neighbor)

        for old_cluster in old_clusters:
            clusters.discard(old_cluster)

        #clusters.update(new_clusters)
        clusterlist = []
        nr_nodes = 0
        for x in clusters:
            if clustergraph.pred[x]:
                height_dif = clustergraph.node[list(clustergraph.pred[x].keys())[0]]["height"]-clustergraph.node[x]["height"]
                node_dif = clustergraph.node[list(clustergraph.pred[x].keys())[0]]["nodes"].difference(clustergraph.node[x]["nodes"])
                clusterlist.append((len(node_dif), height_dif))
            nr_nodes += len(clustergraph.node[x]["nodes"])

        yield height, clusterlist, nr_nodes

    return


def cut_hierarchy_quantile(n, clustergraph, cut_height):
    """ Deprecated version of edge quantile cut. Use DifDenseHDBSCAN instead
    """
    hierarchy_nodes = []

    for (s, t) in clustergraph.edges():
        if clustergraph.node[s]["height"] - clustergraph.node[t]["height"] > cut_height:
            hierarchy_nodes.append((clustergraph.node[t]["nodes"], clustergraph.node[t]["height"]))

    hierarchy_nodes = sorted(hierarchy_nodes, key=itemgetter(1))

    #todo use outlier detection for edge cutting
    ## test with outlier detection
    # edge_lengths = [clustergraph.node[s]["height"] - clustergraph.node[t]["height"] for (s, t) in clustergraph.edges()]
    # [clustergraph.node[v]["height"] for (_, v), b in zip(clustergraph.edges(), is_outlier(np.array(edge_lengths), thresh=50)) if b == True])


    #print("heights", [y for _,y in hierarchy_nodes])
    labels = [-1]*n
    for c_idx in range(len(hierarchy_nodes)):
        for p_idx in hierarchy_nodes[c_idx][0]:
            if labels[p_idx] == -1:
                labels[p_idx] = c_idx
    return labels



def simplify_clustergraph(clustergraph):
    """ remove successive nodes with same set of contained "nodes"

    :param clustergraph: graph to be simplified. Full copy will be created
    :return: returns a new graph which is simplified
    """
    graph = clustergraph.copy()
    start_nodes = [x for (x, y) in graph.in_degree().items() if y == 0]

    edges_to_test = []
    [edges_to_test.extend(graph.edges(node)) for node in start_nodes]

    while edges_to_test:
        (u, v) = edges_to_test.pop(0)

        #clustering does not change
        if graph.node[u]["nodes"] == graph.node[v]["nodes"]:
            for child in graph.edge[v].keys():
                graph.add_edge(u, child)
                edges_to_test.append((u, child))
            #[edges_to_test.extend(graph.edges(node)) for node in graph.edge[v].keys()]
            graph.remove_node(v)
        else:
            edges_to_test.extend(graph.edges(v))

    return graph




if __name__ == "__main__":
    graph = nx.DiGraph()
    graph.add_edges_from([(1, 2), (1, 3), (2, 4), (2, 5), (3, 6), (3, 7), (4, 8), (4,9)])
    graph.node[1]["nodes"] = set([1, 2, 3])
    graph.node[2]["nodes"] = set([1, 2, 3])
    graph.node[3]["nodes"] = set([1, 2])
    graph.node[4]["nodes"] = set([1, 2, 3])
    graph.node[5]["nodes"] = set([1, 2])
    graph.node[6]["nodes"] = set([1])
    graph.node[7]["nodes"] = set([1, 2])
    graph.node[8]["nodes"] = set([1, 2, 3])
    graph.node[9]["nodes"] = set([1, 2])
    new_graph = simplify_clustergraph(graph)

    print(graph.edge)
    print(new_graph.edge)



