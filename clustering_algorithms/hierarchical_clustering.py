from sklearn.cluster import AgglomerativeClustering
from clustering_algorithms.kmeans import KMeans
from clustering_algorithms.minPtsGraphDBScan import minPtsGraphDBScan
import numpy as np


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

    # apply hierarchical cluster
    for linkage in {"single", "complete"}:
        for i in range(2, 40):
            cl = AgglomerativeClustering(n_clusters=i, affinity='precomputed', linkage=linkage)
            cl.fit(sdist_euclidean)
            hom, compl, v = homogeneity_completeness_v_measure(labels_true, cl.labels_)
            print(f"Homogeneity {hom}, \tCompleteness {compl}, \tv-Measure{v}")
            pass
