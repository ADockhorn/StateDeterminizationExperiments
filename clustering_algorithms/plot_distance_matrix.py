
if __name__ == "__main__":
    import os
    os.chdir("\\".join(os.getcwd().split("\\")[:-1]))

    from analysis_tools import load_data_set, normalize_labels, calculate_distance_matrix
    from scipy.spatial.distance import squareform

    # set plot parameters
    HERO_CLASS = "DRUID"
    FUZZY = True
    DISTANCE = "Jaccard"  # either Jaccard or Euclidean

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

    import matplotlib.pyplot as plt
    plt.imshow(sdist_jaccard)

    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().tick_params(color='black', labelcolor='black', bottom=False, left=False, which='both')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.savefig("results_clustering\\distance_matrix_jaccard")
    plt.show()

    plt.imshow(sdist_euclidean)

    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().tick_params(color='black', labelcolor='black', bottom=False, left=False, which='both')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.savefig("results_clustering\\distance_matrix_euclidean")
    plt.show()
