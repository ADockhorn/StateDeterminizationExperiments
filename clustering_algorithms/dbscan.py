import numpy as np
from sklearn.cluster import DBSCAN


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

    for sdist, distance in zip([sdist_jaccard, sdist_euclidean], ["Jaccard", "Euclidean"]):

        eps_values = np.arange(np.min(sdist)+0.001, np.max(sdist),  (np.max(sdist)-np.min(sdist))/20)
        minpts_values = np.arange(2, 22, 1)
        stats = np.zeros((len(eps_values), len(minpts_values), 3))

        for i, eps in enumerate(eps_values):
            for j, minpts in enumerate(minpts_values):
                cl = DBSCAN(eps=eps, min_samples=minpts, metric="precomputed")
                cl.fit(sdist)

                hom, compl, v = homogeneity_completeness_v_measure(labels_true, cl.labels_)
                stats[i, j, :] = [hom, compl, v]
                print(f"eps {eps}, minpts {minpts}\t Homogeneity {hom}, \tCompleteness {compl}, \tv-Measure{v}")

        import matplotlib.pyplot as plt
        import seaborn as sns

        for i, title in enumerate(["homogeneity", "completeness", "v-measure"]):
            plt.figure(figsize=(7, 6))
            chart = sns.heatmap(stats[:, :, i], square=False, xticklabels=minpts_values, cmap="viridis",
                                vmin=0, vmax=1, linewidths=1)
            chart.set_yticklabels(
                chart.get_yticklabels(),
                rotation=45,
                horizontalalignment='right',
                fontweight='light',
            )

            plt.xlabel("$m_{Pts}$", fontsize=14)
            plt.ylabel("$\\varepsilon$-Neighborhood", fontsize=14)
            plt.ylim((0, 20))
            plt.yticks(plt.yticks()[0], ["{:0.2f}".format(val) for val in eps_values][::-1])
            plt.savefig(f"results_clustering\\{title}_{distance.lower()}")
            plt.show()
