from sklearn.cluster import AgglomerativeClustering
from clustering_algorithms.kmeans import KMeans
from clustering_algorithms.minPtsGraphDBScan import minPtsGraphDBScan
import numpy as np
import logging
import time
import matplotlib.pyplot as plt


def plot_results(homogeneity, completeness, vmeasure):

    for alg in homogeneity:
        plt.plot(list(homogeneity[alg].keys()), list(homogeneity[alg].values()), label=alg)
    plt.legend()
    plt.title("homogeneity")
    plt.show()

    for alg in completeness:
        plt.plot(list(completeness[alg].keys()), list(completeness[alg].values()), label=alg)
    plt.legend()
    plt.title("completeness")
    plt.show()

    for alg in vmeasure:
        plt.plot(list(vmeasure[alg].keys()), list(vmeasure[alg].values()), label=alg)
    plt.legend()
    plt.title("vmeasure")
    plt.show()


def plot_results_hierarchical(homogeneity, completeness, vmeasure, alg):
    from matplotlib.cm import get_cmap

    plt.figure(num=None, figsize=(6, 4), dpi=80, facecolor='w', edgecolor='k')
    plt.style.use("bmh")

    plt.gca().tick_params(axis='both', which='major', labelsize=12)
    plt.gca().tick_params(axis='both', which='major', labelsize=12)
    plt.gca().spines['left'].set_visible(True)
    plt.gca().spines['bottom'].set_visible(True)
    plt.gca().spines['left'].set_edgecolor('black')
    plt.gca().spines['bottom'].set_edgecolor('black')
    plt.gca().spines['left'].set_linewidth(1)
    plt.gca().spines['bottom'].set_linewidth(1)
    plt.gca().tick_params(color='black', labelcolor='black', bottom=True, left=True, which='both')
    plt.gca().set_facecolor("w")

    plt.gca().set_frame_on(True)

    plt.gca().xaxis.tick_bottom()
    plt.gca().yaxis.tick_left()

    cmap = [get_cmap("tab20b").colors[12], get_cmap("tab20b").colors[14],
            get_cmap("tab20b").colors[15]]

    n_clusters, hom = list(zip(*[(i, homogeneity[alg][i]) for i in range(10, 260, 10)]))
    compl = [completeness[alg][i] for i in n_clusters]
    vm = [vmeasure[alg][i] for i in n_clusters]
    print(f"best vmeasure {max(vm)}; n_clusters = {n_clusters[np.argmax(vm)]}")

    plt.plot(n_clusters, hom, label="homogeneity", linestyle="solid", marker="D", c=cmap[0], markersize=5)
    plt.plot(n_clusters, compl, label="completeness", linestyle="solid", marker="s", c=cmap[1], markersize=6)
    plt.plot(n_clusters, vm, label="vmeasure", linestyle="dotted", marker="o", c=cmap[2])
    plt.xlabel("#cluster", fontsize=14)
    plt.xticks(range(0, 260, 50), [str(i) if i % 50 == 0 else "" for i in range(0, 260, 50)])
    plt.yticks([0, 0.2, 0.4, 0.6, .8, 1.0])
    legend = plt.legend(fontsize=14, loc='lower right')
    frame = legend.get_frame()
    frame.set_facecolor('w')
    plt.ylim((-0.05, 1.05))
    plt.savefig(f"results_clustering\\{alg}.png")
    plt.show()


def plot_results_kmeans(homogeneity, completeness, vmeasure, alg):
    from matplotlib.cm import get_cmap

    plt.figure(num=None, figsize=(6, 4), dpi=80, facecolor='w', edgecolor='k')
    plt.style.use("bmh")

    plt.gca().tick_params(axis='both', which='major', labelsize=12)
    plt.gca().tick_params(axis='both', which='major', labelsize=12)
    plt.gca().spines['left'].set_visible(True)
    plt.gca().spines['bottom'].set_visible(True)
    plt.gca().spines['left'].set_edgecolor('black')
    plt.gca().spines['bottom'].set_edgecolor('black')
    plt.gca().spines['left'].set_linewidth(1)
    plt.gca().spines['bottom'].set_linewidth(1)
    plt.gca().tick_params(color='black', labelcolor='black', bottom=True, left=True, which='both')
    plt.gca().set_facecolor("w")

    plt.gca().set_frame_on(True)

    plt.gca().xaxis.tick_bottom()
    plt.gca().yaxis.tick_left()

    cmap = [get_cmap("tab20b").colors[12], get_cmap("tab20b").colors[14],
            get_cmap("tab20b").colors[15]]

    n_clusters, hom = list(zip(*[(i, homogeneity[alg][i]) for i in range(10, 200, 10)]))
    compl = [completeness[alg][i] for i in n_clusters]
    vm = [vmeasure[alg][i] for i in n_clusters]
    print(f"best vmeasure {max(vm)}; n_clusters = {n_clusters[np.argmax(vm)]}")

    plt.plot(n_clusters, hom, label="homogeneity", linestyle="solid", marker="D", c=cmap[0], markersize=5)
    plt.plot(n_clusters, compl, label="completeness", linestyle="solid", marker="s", c=cmap[1], markersize=6)
    plt.plot(n_clusters, vm, label="vmeasure", linestyle="dotted", marker="o", c=cmap[2])
    plt.xlabel("#cluster", fontsize=14)
    plt.xticks(range(10, 210, 50), [str(i) if i % 50 == 0 else "" for i in range(10, 210, 50)])
    plt.yticks([0, 0.2, 0.4, 0.6, .8, 1.0])
    legend = plt.legend(fontsize=14, loc='lower right')
    frame = legend.get_frame()
    frame.set_facecolor('w')
    plt.ylim((-0.05, 1.05))
    plt.savefig(f"results_clustering\\{alg}.png")
    plt.show()


def plot_results_dbscan(homogeneity, completeness, vmeasure, alg):
    from matplotlib.cm import get_cmap

    plt.figure(num=None, figsize=(6, 4), dpi=80, facecolor='w', edgecolor='k')
    plt.style.use("bmh")

    plt.gca().tick_params(axis='both', which='major', labelsize=12)
    plt.gca().tick_params(axis='both', which='major', labelsize=12)
    plt.gca().spines['left'].set_visible(True)
    plt.gca().spines['bottom'].set_visible(True)
    plt.gca().spines['left'].set_edgecolor('black')
    plt.gca().spines['bottom'].set_edgecolor('black')
    plt.gca().spines['left'].set_linewidth(1)
    plt.gca().spines['bottom'].set_linewidth(1)
    plt.gca().tick_params(color='black', labelcolor='black', bottom=True, left=True, which='both')
    plt.gca().set_facecolor("w")

    plt.gca().set_frame_on(True)

    plt.gca().xaxis.tick_bottom()
    plt.gca().yaxis.tick_left()

    cmap = [get_cmap("tab20b").colors[12], get_cmap("tab20b").colors[14],
            get_cmap("tab20b").colors[15]]

    n_clusters, hom = list(zip(*[(i, homogeneity[alg][i]) for i in range(0, 100) if i in homogeneity[alg]]))
    compl = [completeness[alg][i] for i in n_clusters]
    vm = [vmeasure[alg][i] for i in n_clusters]
    plt.plot(n_clusters, hom, label="homogeneity", linestyle="solid", marker="D", c=cmap[0], markersize=5)
    plt.plot(n_clusters, compl, label="completeness", linestyle="solid", marker="s", c=cmap[1], markersize=6)
    plt.plot(n_clusters, vm, label="vmeasure", linestyle="dotted", marker="o", c=cmap[2])
    plt.xlabel("#cluster", fontsize=14)
    #plt.xticks(range(0, 210, 50), [str(i) if i % 50 == 0 else "" for i in range(0, 210, 50)])
    plt.yticks([0, 0.2, 0.4, 0.6, .8, 1.0])
    legend = plt.legend(fontsize=14, loc='lower right')
    frame = legend.get_frame()
    frame.set_facecolor('w')
    plt.ylim((-0.05, 1.05))
    plt.savefig(f"results_clustering\\{alg}.png")
    plt.show()


def test_partial_order(sdist_jaccard, sdist_euclidean):
    indices_jaccard = []
    for i in range(len(sdist_jaccard)):
        for j in range(len(sdist_jaccard)):
            indices_jaccard.append((sdist_jaccard[i, j], i, j))

    indices_euclidean = []
    for i in range(len(sdist_euclidean)):
        for j in range(len(sdist_euclidean)):
            indices_euclidean.append((sdist_euclidean[i, j], i, j))

    s1 = sorted(indices_jaccard)
    s2 = sorted(indices_euclidean)

    sorting_inconsistencies = 0
    for i, j in zip(s1, s2):
        if s1[0] != s2[0] or s1[1] != s2[1]:
            sorting_inconsistencies += 1

    print("sorting inconsistencies: ", sorting_inconsistencies)


def save(homogeneity, completeness, vmeasure, labels):
    pickle.dump(homogeneity, open("results_clustering\\homogeneity.txt", "wb"))
    pickle.dump(completeness, open("results_clustering\\completeness.txt", "wb"))
    pickle.dump(vmeasure, open("results_clustering\\vmeasure.txt", "wb"))
    pickle.dump(labels, open("results_clustering\\labels.txt", "wb"))


def load():
    homogeneity = pickle.load(open("results_clustering\\homogeneity.txt", "rb"))
    completeness = pickle.load(open("results_clustering\\completeness.txt", "rb"))
    vmeasure = pickle.load(open("results_clustering\\vmeasure.txt", "rb"))
    labels = pickle.load(open("results_clustering\\labels.txt", "rb"))
    return homogeneity, completeness, vmeasure, labels


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

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
    logging.info("loading the deck data sets: {} s".format(end_time - start_time))


    # calculate distance matrices
    start_time = time.time()
    dist_jaccard = calculate_distance_matrix(playedDecks, measure="jaccard")
    sdist_jaccard = squareform(dist_jaccard)
    dist_euclidean = calculate_distance_matrix(playedDecks, measure="euclidean")
    sdist_euclidean = squareform(dist_euclidean)
    #test_partial_order()
    end_time = time.time()
    logging.info("calculation of distance matrixes: {} s".format(end_time - start_time))

    homogeneity = {"single": {}, "complete": {}, "kmeans": {}, "dbscan-2": {}, "dbscan-5": {}, "dbscan-10": {},
                   "single-e": {}, "complete-e": {}, "kmeans-e": {}, "dbscan-2-e": {}, "dbscan-5-e": {}, "dbscan-10-e": {}}
    completeness = {"single": {}, "complete": {}, "kmeans": {}, "dbscan-2": {}, "dbscan-5": {}, "dbscan-10": {},
                    "single-e": {}, "complete-e": {}, "kmeans-e": {}, "dbscan-2-e": {}, "dbscan-5-e": {}, "dbscan-10-e": {}}
    vmeasure = {"single": {}, "complete": {}, "kmeans": {}, "dbscan-2": {}, "dbscan-5": {}, "dbscan-10": {},
                "single-e": {}, "complete-e": {}, "kmeans-e": {}, "dbscan-2-e": {}, "dbscan-5-e": {}, "dbscan-10-e": {}}
    labels = {"single": {}, "complete": {}, "kmeans": {}, "dbscan-2": {}, "dbscan-5": {}, "dbscan-10": {},
              "single-e": {}, "complete-e": {}, "kmeans-e": {}, "dbscan-2-e": {}, "dbscan-5-e": {}, "dbscan-10-e": {}}

    # apply hierarchical cluster jaccard
    for linkage in {"single", "complete"}:
        start_time = time.time()

        for i in range(2, 956):
            cl = AgglomerativeClustering(n_clusters=i, affinity='precomputed', linkage=linkage)
            cl.fit(sdist_jaccard)
            hom, compl, v = homogeneity_completeness_v_measure(labels_true, cl.labels_)
            homogeneity[linkage][i] = hom
            completeness[linkage][i] = compl
            vmeasure[linkage][i] = v
            labels[linkage][i] = cl.labels_

            logging.debug(f"{linkage}: Homogeneity {hom}, \tCompleteness {compl}, \tv-Measure{v}")
            pass
        end_time = time.time()
        logging.info(f"calculation of HAC ({linkage}-linkage): {end_time - start_time} s")

    # apply hierarchical cluster euclid
    for linkage in {"single", "complete"}:
        start_time = time.time()

        for i in range(2, 956):
            cl = AgglomerativeClustering(n_clusters=i, affinity='precomputed', linkage=linkage)
            cl.fit(sdist_euclidean)
            hom, compl, v = homogeneity_completeness_v_measure(labels_true, cl.labels_)
            homogeneity[linkage+"-e"][i] = hom
            completeness[linkage+"-e"][i] = compl
            vmeasure[linkage+"-e"][i] = v
            labels[linkage+"-e"][i] = cl.labels_

            logging.debug(f"{linkage}-e: Homogeneity {hom}, \tCompleteness {compl}, \tv-Measure{v}")
            pass
        end_time = time.time()
        logging.info(f"calculation of HAC ({linkage}-linkage): {end_time - start_time} s")


    # apply DBSCAN Jaccard
    for i in [2, 5, 10]:

        start_time = time.time()
        db = minPtsGraphDBScan(minPts=i)
        db.fit(playedDecks, sdist_jaccard)
        db.cut_height_statistics()

        label_gen = db.cut_height_label_gen()
        lowest_n = 1000
        best_vmeasure = 0
        for idx, labels_ in enumerate(label_gen):
            hom, compl, v = homogeneity_completeness_v_measure(labels_true, labels_[1])
            n_clusters = len(set(labels_[1]))
            if n_clusters <= lowest_n:
                homogeneity[f"dbscan-{i}"][n_clusters] = hom
                completeness[f"dbscan-{i}"][n_clusters] = compl
                vmeasure[f"dbscan-{i}"][n_clusters] = v
                labels[f"dbscan-{i}"][n_clusters] = labels_[1]
                lowest_n_n = n_clusters

                logging.debug(f"dbscan-{i}: label_idx {idx},\t Homogeneity {hom}, \tCompleteness {compl}, \tv-Measure{v}")
            if v > best_vmeasure:
                best_vmeasure = v
                print("height: ", labels_[0], "vmeasure: ", v)
        end_time = time.time()
        logging.info(f"calculation of DBSCAN-{i}: {end_time - start_time} s")


    # apply DBSCAN Euclid
    for i in [2, 5, 10]:

        start_time = time.time()
        db = minPtsGraphDBScan(minPts=i)
        db.fit(playedDecks, sdist_euclidean)
        db.cut_height_statistics()

        label_gen = db.cut_height_label_gen()
        lowest_n = 1000
        for idx, labels_ in enumerate(label_gen):
            hom, compl, v = homogeneity_completeness_v_measure(labels_true, labels_[1])
            n_clusters = len(set(labels_[1]))
            if n_clusters <= lowest_n:
                homogeneity[f"dbscan-{i}-e"][n_clusters] = hom
                completeness[f"dbscan-{i}-e"][n_clusters] = compl
                vmeasure[f"dbscan-{i}-e"][n_clusters] = v
                labels[f"dbscan-{i}-e"][n_clusters] = labels_[1]
                lowest_n_n = n_clusters

            logging.debug(f"dbscan-{i}: label_idx {idx},\t Homogeneity {hom}, \tCompleteness {compl}, \tv-Measure{v}")
        end_time = time.time()
        logging.info(f"calculation of DBSCAN-{i}: {end_time - start_time} s")

    # apply k-means
    start_time = time.time()
    clusters = np.arange(10, 260, 10)
    stats = np.zeros((len(clusters), 3))
    for idx, i in enumerate(clusters):
        if i in homogeneity["kmeans"]:
            continue

        cl = KMeans(n_clusters=i, n_starts=10)
        cl.fit(playedDecks)

        hom, compl, v = homogeneity_completeness_v_measure(labels_true, cl.labels_)
        homogeneity["kmeans"][i] = hom
        completeness["kmeans"][i] = compl
        vmeasure["kmeans"][i] = v
        labels["kmeans"][i] = cl.labels_

        stats[idx, :] = [hom, compl, v]
        logging.info(f"kmeans: n_clusters {i},\t Homogeneity {hom}, \tCompleteness {compl}, \tv-Measure{v}")


    end_time = time.time()
    logging.info(f"calculation of K-Means: {end_time - start_time} s")


    import pickle
    homogeneity, completeness, vmeasure, labels = load()

    plot_results_hierarchical(homogeneity, completeness, vmeasure, "single")
    plot_results_hierarchical(homogeneity, completeness, vmeasure, "complete")
    plot_results_kmeans(homogeneity, completeness, vmeasure, "kmeans")
    plot_results_dbscan(homogeneity, completeness, vmeasure, "dbscan-2")
    plot_results_dbscan(homogeneity, completeness, vmeasure, "dbscan-5")
    plot_results_dbscan(homogeneity, completeness, vmeasure, "dbscan-10")

    save(homogeneity, completeness, vmeasure, labels)
