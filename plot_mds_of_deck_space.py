if __name__ == "__main__":

    from analysis_tools import load_data_set, normalize_labels, calculate_distance_matrix
    from scipy.spatial.distance import squareform

    # set plot parameters
    HERO_CLASS = "HUNTER"
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

    import json
    with open("deck_data\\Archetypes.json") as f:
        data = json.load(f)
    archetypes = {entry["id"]: entry["name"] for entry in data}
    archetypes[-3] = "Unknown"

    # 2D projection of the clustered deck space
    from sklearn.manifold import MDS
    import seaborn as sns
    from matplotlib.colors import ListedColormap
    import matplotlib.pyplot as plt

    embedding = MDS(n_components=2, dissimilarity="precomputed", random_state=17041991)
    deck_transformed = embedding.fit_transform(sdist_euclidean)

    cmap = ListedColormap(sns.color_palette("deep", len(set(labels_true))))

    plt.style.use("bmh")
    fig=plt.figure(figsize=(10,6))
    plt.scatter(deck_transformed[:, 0], deck_transformed[:, 1], c=labels_true, cmap=cmap)

    plt.gca().tick_params(axis='both', which='major', labelsize=12)
    plt.gca().tick_params(axis='both', which='major', labelsize=12)
    plt.gca().spines['left'].set_visible(True)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
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

    plt.xlabel("Embedded Component 1", fontsize=14)
    plt.ylabel("Embedded Component 2", fontsize=14)

    for label, legend_entry in {label: archetypes[archetype_label_dict[label]] for label in labels_true}.items():
        dummy = plt.gca().plot([], [], ls='-', c=sns.color_palette("deep", len(set(labels_true)))[label], label=legend_entry)[0]
    legend = plt.gca().legend(loc='upper left', bbox_to_anchor=(1, 1.0), fontsize=12)
    frame = legend.get_frame()
    frame.set_facecolor('w')
    frame.set_edgecolor('w')
    plt.show()
