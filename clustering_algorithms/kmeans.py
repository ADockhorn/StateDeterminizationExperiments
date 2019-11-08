import random
import numpy as np
from clustering_algorithms.fuzzy_deck_cluster import FuzzyDeckCluster


def calculate_pairwise_euclidean_distance(centers, deck_data):
    dist_mat = np.zeros((len(centers), len(deck_data)))
    for center_idx, center in enumerate(centers):
        for deck_idx, deck in enumerate(deck_data):
            dist_mat[center_idx, deck_idx] = center.euclidean_distance(deck)
    return dist_mat


def calculate_pairwise_jaccard_distance(centers, deck_data):
    dist_mat = np.zeros((len(centers), len(deck_data)))
    for center_idx, center in enumerate(centers):
        for deck_idx, deck in enumerate(deck_data):
            dist_mat[center_idx, deck_idx] = center.jaccard_distance(deck)
    return dist_mat


class KMeans:

    def __init__(self, n_clusters=None, n_starts=100, distance="jaccard"):
        self.centers = None
        self.labels_ = None
        self.n_clusters = n_clusters
        self.n_starts = n_starts
        self.n_data_points = -1
        self.iterations = 0
        if distance not in {"jaccard", "euclidean"}:
            raise ValueError(f"distance needs to be 'jaccard' or 'euclidean', but is {distance}")
        else:
            self.distance = distance

    def fit(self, deck_data):
        self.n_data_points = len(deck_data)
        best_sse = float("inf")
        for i in range(self.n_starts):
            centers, labels = self.cluster(deck_data)
            sse = self.determine_sse(centers, labels, deck_data)
            if sse < best_sse:
                best_sse = sse
                self.labels_ = labels
                self.centers = centers

    def cluster(self, deck_data):
        tmp_centers = random.sample(deck_data, self.n_clusters)
        deck_data = np.array(deck_data)

        convergence = False
        previous_labels = np.array([-1]*len(deck_data))
        tmp_labels = None

        while not convergence:
            tmp_labels = self.assign_clusters(tmp_centers, deck_data)
            tmp_centers = self.move_cluster_centers(deck_data, tmp_labels)

            if np.sum(previous_labels == tmp_labels) != len(deck_data):
                previous_labels = tmp_labels
            else:
                convergence = True
        return tmp_centers, tmp_labels

    def move_cluster_centers(self, deck_data, labels):
        centers = []
        for i in range(self.n_clusters):
            centers.append(FuzzyDeckCluster(deck_data[i == labels]).centroid())
        return centers

    def assign_clusters(self, centers, deck_data):
        if self.distance == "jaccard":
            dist_mat = calculate_pairwise_jaccard_distance(centers, deck_data)
        else:
            dist_mat = calculate_pairwise_euclidean_distance(centers, deck_data)

        return np.argmin(dist_mat, axis=0)

    def determine_sse(self, centers, labels, deck_data):
        if self.distance == "jaccard":
            dist_mat = calculate_pairwise_jaccard_distance(centers, deck_data)
        else:
            dist_mat = calculate_pairwise_euclidean_distance(centers, deck_data)

        sse_sum = 0.0
        for column_idx, row_idx in enumerate(labels):
            sse_sum += (dist_mat[row_idx, column_idx] ** 2)
        return sse_sum


if __name__ == "__main__":
    import os
    os.chdir("\\".join(os.getcwd().split("\\")[:-1]))

    from analysis_tools import load_data_set, normalize_labels, calculate_distance_matrix
    from scipy.spatial.distance import squareform
    from sklearn.metrics import homogeneity_completeness_v_measure

    # set plot parameters
    HERO_CLASS = "ALL"
    FUZZY = True

    # load data set
    playedDecks = load_data_set(HERO_CLASS, FUZZY, "deck_data/Decks.json")
    id_to_index = {deck_id: i for i, deck_id in enumerate([p.deck_id for p in playedDecks])}
    MAX_N = len(playedDecks)
    archetype_label_dict, labels_true = normalize_labels(labels=[d.archetype[0] for d in playedDecks])

    # calculate distance matrices
    dist_jaccard = calculate_distance_matrix(playedDecks, measure="jaccard")
    sdist_jaccard = squareform(dist_jaccard)
    dist_euclidean = calculate_distance_matrix(playedDecks, measure="euclidean")
    sdist_euclidean = squareform(dist_euclidean)

    clusters = np.arange(1, 80, 4)
    stats = np.zeros((len(clusters), 3))

    for idx, i in enumerate(clusters):
        cl = KMeans(n_clusters=i, n_starts=3)
        cl.fit(playedDecks)

        hom, compl, v = homogeneity_completeness_v_measure(labels_true, cl.labels_)
        stats[idx, :] = [hom, compl, v]
        print(f"n_clusters {i},\t Homogeneity {hom}, \tCompleteness {compl}, \tv-Measure{v}")
        pass

    import matplotlib.pyplot as plt
    plt.plot(clusters, stats[:, 0], label="Homogeneity")
    plt.plot(clusters, stats[:, 1], label="Completeness")
    plt.plot(clusters, stats[:, 2], label="VMeasure")
    plt.legend()
    plt.show()
