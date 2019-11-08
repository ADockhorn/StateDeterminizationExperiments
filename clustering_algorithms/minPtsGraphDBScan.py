__author__ = 'Alex'

from sklearn.metrics import pairwise_distances
import time
import networkx as nx
from clustering_algorithms.GraphTools import connected_component_subgraphs_core
from clustering_algorithms.RandSelect import *
import logging
import numpy as np


class minPtsGraphDBScan():
    """ Minimum Spanning Tree implementation of minPtsDBSCAN.
    Building a hierarchy of valid eps values for a fixed minPts.

    Used distance measure:
        d(i,j) = max(min(d_core(i), d_core(j)), d(i, j))
    """

    def __init__(self, minPts):
        self.configure(minPts)


    def configure(self, minPts):
        """ setup of the cluster object for the next run of fit()

        :param minPts: fix minPts
        """
        self.minPts = minPts
        self.clustered_dataset = None
        self.clustergraph = None
        self.last_prediction_type = None
        self.simplified_clustergraph = None


    def simplify_clustergraph(self):
        """ nothing todo here, just for compatibility with minPtsHierarchicalDBScan

        :return: does nothing, returns nothing
        """
        return


    def get_nr_of_clusterlevels(self):
        """ get an upper bound estimate of the number of clusterings

        The number can be lower, when the clustering doesn't change by adding an edge to the spanning tree.
        For example, when a border point becomes core, but all the points in its neighborhood are already
        part of the cluster.

        :return: number of hierarchy levels
        """
        return len(self.get_clusterlevels())


    def get_clusterlevels(self):
        """ get all possible eps values in which the clustering might change.

        The number can be lower, when the clustering doesn't change by adding an edge to the spanning tree.
        For example, when a border point becomes core, but all the points in its neighborhood are already
        part of the cluster.

        :return: set of hierarchy levels
        """
        return set([self.clustergraph.edge[u][v]["weight"] for u, v in self.clustergraph.edges()])



    def predict_cut_height(self, height):
        """ return labels of a single horizontal dendrogram cut

        :param "cut_height" : removes all edges larger then height (*args = )
        :return: labellist
        """
        graph = self.clustergraph.copy()
        for u, v in graph.edges():
            if graph.edge[u][v]["weight"] >= height:
                graph.remove_edge(u, v)

        labels = [-1]*len(graph)

        cluster_idx = 0
        clusterheights={}
        for graph in connected_component_subgraphs_core(graph, max_core_distance=height):
            if len(graph) <= 1:
                continue
            for node in graph.nodes():
                if labels[node] == -1:
                    labels[node] = cluster_idx
                else:
                    labels[node] = -2
            #print("edges:", [graph.edge[u][v]["weight"] for u, v in graph.edges()])
            clusterheights[cluster_idx] = max([graph.edge[u][v]["weight"] for u, v in graph.edges()])
            cluster_idx +=1
        return labels, clusterheights



    def cut_height_label_gen(self, detailed_hierarchy=True):
        """ Generator method for generating or filtering levels of the hierarchy

        :param: detailed_hierarchy: if False apply filtering of the cluster levels,
        dependent on the actions done inbetween
        :return: generator for filtered or unfiltered hierarchy
        """
        change_count = {"new core" : 0, "merge" : 0, "new node" : 0, "new core merge":0, "new single core":0}
        gen = self.cut_height_label_gen_unfiltered(change_count)
        last_change_sum = change_count["new core merge"] + change_count["merge"]
        reported = 0
        seen = 0
        report_next = False

        last = None
        if detailed_hierarchy:
            for (height, labels, clusterheight) in gen:
                seen += 1
                reported += 1
                yield (height, labels, clusterheight)
        else:
            for (height, labels, clusterheight) in gen:
                if last is None:
                    last = (height, labels, clusterheight.copy())
                seen += 1
                if change_count["new core merge"] + change_count["merge"] > last_change_sum or report_next:
                    report_next = False
                    if change_count["new core merge"] + change_count["merge"] > last_change_sum:
                        report_next = True
                    last_change_sum =  change_count["new core merge"] + change_count["merge"]
                    reported += 1
                    yield last
                last = (height, labels, clusterheight.copy())

        logging.info("seen {}, reported {}".format(seen, reported))


    def cut_height_statistics(self):
        """ report the number of changes per type

        :return: number of new cores, normal merges, new nodes, and new core merges
        """
        change_count = {"new core" : 0, "merge" : 0, "new node" : 0, "new core merge":0, "new single core":0}
        gen = self.cut_height_label_gen_unfiltered(change_count)
        for (height, labels, clusterheight) in gen:
            continue
        return change_count


    def cut_height_label_gen_unfiltered(self, change_count={"new core" : 0, "merge" : 0, "new node" : 0, "new core merge":0}):
        """ get all differing labelings for possible eps values.

        :yield: eps, labeling
        """
        sorted_list = sorted([(self.clustergraph.get_edge_data(u, v)["weight"], (u, v)) for u, v in self.clustergraph.edges()])

        # use horizontal and vertical structure for faster processing
        cluster_id = 0                                          #eindeutige cluster id
        labels = {x: set() for x in self.clustergraph}          #initialize everything as noise
        current_clusters = {}                                   #list of nodesets
        clusterheight = {}

        current_height = sorted_list[0][0]              #initialization = minimal weight

        new_cores = []
        cluster_merges = []
        simple_edge = []

        #iterate through all edges of the minimal spanning tree
        for (height, (u,v)) in sorted_list + [(np.inf, (-1, -1))]:
            if u > v:
                continue    #avoid simetry of edges

            if height > current_height:
                changes = False
                #process new cores:
                for core in new_cores:
                    #
                    # if core == 0:
                    #    print()
                    current_clusters[cluster_id] = set([core])   # create new cluster for merging
                    clusterheight[cluster_id] = current_height
                    if len(labels[core]) <= 1:
                        changes = True                      # node was borderpoint -> labeling will be a merge of the last
                        change_count["new core"] += 1
                    if len(labels[core]) > 1:
                        changes = True                      # node was borderpoint -> labeling will be a merge of the last
                        change_count["new core merge"] += 1

                    for cluster in labels[core]:                # merge all clusters core is in and add their points to new cluster
                                                                # will be skipped, when point was noise before (only for minPts < 3)
                        for point in current_clusters[cluster]: # remove cluster from points setlist and add new clusterid
                            if point == core:
                                continue
                            labels[point].remove(cluster)       # remove cluster point belonged to
                            labels[point].add(cluster_id)       # add new merging cluster
                        current_clusters[cluster_id].update(current_clusters[cluster])
                        current_clusters.pop(cluster)           # throw away old cluster since it doesn't appear in a points setlist anymore
                        clusterheight.pop(cluster)
                    clusterheight[cluster_id] = current_height
                    labels[core] = set([cluster_id])
                    cluster_id += 1

                #process cluster merge
                for (a, b) in cluster_merges:
                    current_clusters[cluster_id] = set([a,b])   # create new cluster for merging
                    clusterheight[cluster_id] = current_height
                    combined_labels = labels[a] | labels[b]
                    if len(combined_labels) > 1:
                        changes = True                          # node belong to different clusters -> labeling will be a merge of the last
                        change_count["merge"] += 1

                    for cluster in combined_labels:             # merge all clusters u is in and add their points to new cluster
                                                                # will be skipped, when u was noise before (only for minPts < 3)

                        for point in current_clusters[cluster]: # remove cluster from points setlist and add new clusterid
                            if point == a or point == b:
                                continue
                            labels[point].remove(cluster)       # remove cluster u belonged to
                            labels[point].add(cluster_id)       # add new merging cluster
                        current_clusters[cluster_id].update(current_clusters[cluster])
                        current_clusters.pop(cluster)           # throw away old cluster since it doesn't appear in a points setlist anymore
                        clusterheight.pop(cluster)
                    labels[a] = set([cluster_id])
                    labels[b] = set([cluster_id])
                    clusterheight[cluster_id] = current_height
                    cluster_id += 1
                cluster_merges = []

                #process simple edges
                for (a, b) in simple_edge:
                    changes = True
                    change_count["new node"] += 1

                    if self.clustergraph[a][a]["weight"] <= self.clustergraph[b][b]["weight"]:
                        #add v to u
                        source = a
                        target = b
                    else :
                        #add u to v
                        source = b
                        target = a
                    new_cores = [x for x in new_cores if x != source]

                    labels[target].update(labels[source])                 #u can still be border point of other clusters
                    for cluster in labels[source]:
                        current_clusters[cluster].update([target])   #core distance edge could be processed after this
                        clusterheight[cluster] = current_height
                simple_edge = []


                if new_cores:
                    change_count["new single core"] += len(new_cores)
                    change_count["new core"] -= len(new_cores)
                new_cores = []


                #print(labels)
                if changes:
                    current_labeling = [-1 if not labels[x] else -2 if len(labels[x])>1 else list(labels[x])[0]
                                        for x in self.clustergraph]
                    yield current_height, current_labeling, clusterheight

                current_height = height

            if u == v:      #core distance reached
                new_cores.append(u)

            else:
                if self.clustergraph[u][u]["weight"] <= height and self.clustergraph[v][v]["weight"] <= height:
                    cluster_merges.append((u,v))
                else:
                    simple_edge.append((u,v))


    def get_clusterings(self):
        """ Generator function. yields the clustering of each dendrogram level

        :yield: "level, labels, clusterheights": cut-height, labellist, height of each cluster
        """
        #levels = sorted(list(self.get_clusterlevels()), reverse=True)
        #for level in levels:
            #labels, clusterheights = self.predict_cut_height(level)
            #yield level, labels, clusterheights
        label_gen = self.cut_height_label_gen()
        for level in label_gen:
            yield level


    #todo not implemented yet
    def predict_cut_edge_quantile(self, clustergraph, height):
        #sort numerated edge weights
        #calculate difference to last edge
        #cut edge with the heighest differences
        return None


    def get_core_distance(self, distance_list, minPts, n):
        """ Returns the distance of the k-th nearest neighbor

        :param "distance_list": pairwise distances of the point to all points
        :param "minPts" : minimum number of points of the neighborhood set
        :param "n": number of points in the dataset

        :reutnr: binning distance matrix into equal sized bins in the range of (0, max_range)
        """
        return RandSelect(distance_list, minPts, n)


    def fit(self, dataset, dist_mat = None, max_range = None, bins = None):
        """ Generates the minimun spanning tree for a dbscan clustering based on a fixed minPts.
        Self edges are included to represent d_core(i).

        :param "dataset": dataset to be clusteres, needed for the plot
        :param "dist_mat" : optional distance matrix, default: euclidean distance will be calculated
        :param "max_range": maximal epsilon value that shozld be included, this can considerably speed up the process
        :param "bin": binning distance matrix into equal sized bins in the range of (0, max_range)
        """

        logging.info("creation of hierarchy for minPts = {}".format(self.minPts))

        N = len(dataset)
        # store dataset, necessary for later plotting
        self.dataset = dataset

        # calculate distance matrix if not given by the user
        if dist_mat is None:
            distance_matrix = pairwise_distances(dataset)
        else:
            distance_matrix = dist_mat

        logging.info("calculate core distances")
        core_distance = [self.get_core_distance(distance_matrix[x,:], self.minPts, N) for x in range(N)]
        logging.debug("core_distances: {}".format(core_distance))

        # set max range
        if max_range is None:
            max_range = np.Inf
            # other valid estimate, but does not generate the whole tree
            #max(self.core_distance)
        logging.debug("max_range: {}".format(max_range))

        logging.info("build graph structure")
        graph = nx.empty_graph(N)
        # iterate through distance matrix and build graph
        for source in range(N):
            for target in range(source+1,N):
                dist = max(min(core_distance[source], core_distance[target]), distance_matrix[source, target])
                #dist = min(max(self.core_distance[source], distance_matrix[source, target]), max(distance_matrix[source, target], self.core_distance[target]))
                #dist = max(max(self.core_distance[source], distance_matrix[source, target]), max(distance_matrix[source, target], self.core_distance[target]))
                if max_range >= dist:
                    graph.add_edge(source, target, weight=dist)

        # calculate minimum spanning tree
        logging.info("create spanning tree")
        self.clustergraph = nx.minimum_spanning_tree(graph)

        # add self edges as core distances
        self.clustergraph.add_edges_from([(i, i, {"weight": core_distance[i]} )for i in range(len(core_distance))])

    def plot(self, cut_height):
        """ Plots the DBSCAN minimum spanning tree of the dataset, without edges larger than cut_height

        :param cut_height: maximal edge weight of the graph
        :return:
        """
        # plot clustering of the given cut_height
        plt.subplot(2,1,2)
        labels, _= self.predict_cut_height(cut_height)
        plt.scatter(dataset[:,0], dataset[:,1], c=labels, s = 100)
        plt.title("height {}".format(cut_height))
        plt.gca().axis("equal")
        xlimit = plt.xlim()
        ylimit = plt.ylim()

        # plot minimum spanning tree
        graph = self.clustergraph.copy()
        for u, v in graph.edges():
            if graph.edge[u][v]["weight"] >= cut_height:
                graph.remove_edge(u, v)

        n = len(dataset)
        plt.subplot(2,1,1)
        position = {x: (self.dataset[x, 0], self.dataset[x, 1]) for x in range(n)}
        #nx.draw_networkx_edges(self.clustergraph, position)
        nx.draw_networkx_edges(graph, position)
        plt.gca().axis("equal")

        plt.xlim(xlimit)
        plt.ylim(ylimit)

        plt.show()



def renumber_labels(labeling):
    """ Utility function for the relabeling of cluster labels

    :param labeling: original clustering
    :return: relabeled clustering
    """
    unique_labels = list(set(labeling) - set([-1, -2]))
    translation_dict = {unique_labels[x]: x for x in range(len(unique_labels))}
    translation_dict[-1] = -1
    translation_dict[-2] = -2
    return [translation_dict[x] for x in labeling]


def dendrogram_distribution(dataset, max_k=20):
    """ returns the statistic of edge types in the minimum spanning tree

    :param "max_k": creates statistic for minPts from 2 to max_k
    :return: returns ["k", "new node", "new core", "new core merge", "merge"]
    """
    statistics = np.zeros(((max_k-1)), dtype=[("k", 'i4'),("new node", 'i4'),("new core", 'i4'),
                                           ("new core merge", 'i4'),("merge", 'i4')])
    for k in range(2, max_k+1):
        db = minPtsGraphDBScan(minPts=k)
        distance_matrix = pairwise_distances(dataset)
        db.fit(dataset, distance_matrix, max_range=2)
        change_count = db.cut_height_statistics()
        change_count["new core"] += change_count["new single core"]
        statistics[k-2] = np.array([k, change_count["new node"], change_count["new core"],
                               change_count["new core merge"], change_count["merge"]])
    return statistics


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    np.random.seed(17041991)

    import os
    os.chdir("\\".join(os.getcwd().split("\\")[:-1]))

    from analysis_tools import load_data_set, normalize_labels, calculate_distance_matrix
    from scipy.spatial.distance import squareform
    from sklearn.metrics import homogeneity_completeness_v_measure

    # set plot parameters
    HERO_CLASS = "ALL"
    FUZZY = True

    # load data set
    start_time = time.time()
    playedDecks = load_data_set(HERO_CLASS, FUZZY, "deck_data/Decks.json")
    id_to_index = {deck_id: i for i, deck_id in enumerate([p.deck_id for p in playedDecks])}
    MAX_N = len(playedDecks)
    archetype_label_dict, labels_true = normalize_labels(labels=[d.archetype[0] for d in playedDecks])
    end_time = time.time()
    logging.info("loading the deck data sets: {} s".format(end_time-start_time))

    # calculate distance matrices
    start_time = time.time()
    dist_jaccard = calculate_distance_matrix(playedDecks, measure="jaccard")
    sdist_jaccard = squareform(dist_jaccard)
    dist_euclidean = calculate_distance_matrix(playedDecks, measure="euclidean")
    sdist_euclidean = squareform(dist_euclidean)
    end_time = time.time()
    logging.info("calculation of distance matrixes: {} s".format(end_time-start_time))

    start_time = time.time()
    db = minPtsGraphDBScan(minPts=2)
    db.fit(playedDecks, sdist_jaccard)
    db.cut_height_statistics()
    end_time = time.time()
    logging.info("calculation of all clusters: {} s".format(end_time-start_time))

    start_time = time.time()
    homogeneity = []
    completeness = []
    vmeasure = []
    label_gen = db.cut_height_label_gen()
    for idx, labels in enumerate(label_gen):
        hom, compl, v = homogeneity_completeness_v_measure(labels_true, labels[1])
        homogeneity.append(hom)
        completeness.append(compl)
        vmeasure.append(v)
        print(f"label_idx {idx},\t Homogeneity {hom}, \tCompleteness {compl}, \tv-Measure{v}")
    end_time = time.time()
    logging.info("evaluation of homogeneity, completeness and vmeasure: {} s".format(end_time-start_time))

    import matplotlib.pyplot as plt
    plt.plot(range(len(homogeneity)), homogeneity, label="Homogeneity")
    plt.plot(range(len(completeness)), completeness, label="Completeness")
    plt.plot(range(len(vmeasure)), vmeasure, label="VMeasure")
    plt.legend()
    plt.show()
    print("test")
