import json
import numpy as np
import matplotlib.pyplot as plt
from bigram_extraction import *
import time


def load_dataset(filenames, minimal_length):
    dataset = []
    for file in filenames:
        with open(file) as json_file:
            data = json.load(json_file)
            dataset.extend([game for game in data["games"]
                            if max([0] + [action["turn"] for action in game["card_history"]]) >= minimal_length])
    return dataset


def get_turn_dicts(game):
    cards_per_turn_player = dict()
    cards_per_turn_opponent = dict()
    for action in game["card_history"]:
        target_dict = cards_per_turn_player if action["player"] == "me" else cards_per_turn_opponent
        if action["turn"] in target_dict:
            target_dict[action["turn"]].append(action["card"]["id"])
        else:
            target_dict[action["turn"]] = [action["card"]["id"]]
    return cards_per_turn_player, cards_per_turn_opponent


def evaluate_top_k_predictions_remaining_game(turn, predictions, cards_per_turn):
    correct = [False] * len(predictions)

    for pred_idx, (prediction, prediction_count) in enumerate(predictions):

        for turn_idx in range(turn + 1, max(cards_per_turn)+1):
            if turn_idx in cards_per_turn:
                if prediction in cards_per_turn[turn_idx]:
                    correct[pred_idx] = True
                    break

    return correct


def evaluate_top_k_predictions_next_turn(turn, predictions, cards_per_turn):
    correct = [False] * len(predictions)

    for pred_idx, (prediction, prediction_count) in enumerate(predictions):

        turn_idx = turn + 1
        if turn_idx in cards_per_turn:
            if prediction in cards_per_turn[turn_idx]:
                correct[pred_idx] = True
                break

    return correct


def evaluate_top_k_predictions_turn_dict(cluster_centroids, cards_per_turn, evaluate_turns, k, filter=False,
                                         only_next_turn=False, calculate_any_k=False):
    from clustering_algorithms.fuzzy_deck_cluster import FuzzyDeck
    prediction = dict()
    statistics = np.zeros((evaluate_turns, k+1, 2))
    observed_cards = set()

    if not cards_per_turn:
        return statistics

    for turn in range(1, min(max(cards_per_turn)-1, evaluate_turns+1)):

        if turn in cards_per_turn:
            for card in cards_per_turn[turn]:
                observed_cards.add(card)
            fuzzy_deck = FuzzyDeck(observed_cards)
            pass

        if filter:
            prediction_tuples = sorted([(x, y) for x, y in prediction.items() if x not in observed_cards],
                                       key=lambda tup: tup[1], reverse=True)[:k]
        else:
            prediction_tuples = sorted([(x, y) for x, y in prediction.items()],
                                       key=lambda tup: tup[1], reverse=True)[:k]

        if not only_next_turn:
            correct = evaluate_top_k_predictions_remaining_game(turn, prediction_tuples, cards_per_turn)
        else:
            correct = evaluate_top_k_predictions_next_turn(turn, prediction_tuples, cards_per_turn)

        if calculate_any_k:
            for i in range(k):
                if True in correct[:i]:
                    if len(correct) <= i:
                        correct.append(True)
                    else:
                        correct[i] = True

        for idx, entry in enumerate(correct):
            if entry:
                statistics[turn-1, idx, 0] += 1
            else:
                statistics[turn-1, idx, 1] += 1

        if True in correct:
            statistics[turn - 1, -1, 0] += 1
        else:
            statistics[turn - 1, -1, 1] += 1

    return statistics


def plot_basics():
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


def plot_statistics_per_top_k(statistics, bigram_name, k, title, filename=None):
    plot_basics()

    plt.plot(np.mean(np.array([statistics[:, i, 0] / (statistics[:, i, 0] + statistics[:, i, 1]) for i in range(0, k)]),
                     axis=0), label=f"Average Prediction Rank 1 - {k}", linestyle="dashed", alpha=0.5)

    plt.plot(statistics[:, 0, 0] / (statistics[:, 0, 0] + statistics[:, 0, 1]), label="Prediction Rank 1",
             marker="^", markeredgecolor='none', color="g", linestyle='')

    """
    for i in range(1, k-2):
        plt.plot(statistics[:, i, 0] / (statistics[:, i, 0] + statistics[:, i, 1]),
                 marker="s", markeredgecolor='none', color="b", linestyle='')

    plt.plot(statistics[:, k-2, 0] / (statistics[:, k-2, 0] + statistics[:, k-2, 1]),
             label=f"Prediction ranked from 2 to {k-1}",
             marker="s", markeredgecolor='none', color="b", linestyle='')
    """

    plt.plot(statistics[:, k - 1, 0] / (statistics[:, k - 1, 0] + statistics[:, k - 1, 1]), label=f"Prediction Rank {k}",
             marker="v", markeredgecolor='none', color="r", linestyle='')

    plt.xlabel("Turn", fontsize=14)
    plt.xticks(np.arange(10), [str(i) for i in range(1, 11)])

    plt.ylabel("Prediction Accuracy", fontsize=14)
    plt.ylim((0, 1))
    plt.yticks(np.arange(0, 1.01, 0.25), ["{:0.2f}".format(i) for i in np.arange(0, 1.01, 0.25)])

    handles, labels = plt.gca().get_legend_handles_labels()
    order = [1, 0, 2]
    legend = plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
    frame = legend.get_frame()
    frame.set_facecolor('w')

    #plt.title(title, fontsize=16, pad=10)
    if filename:
        plt.savefig(f'{result_folder}/{filename}.png')
    plt.show()


def plot_statistics_any_top_k(statistics_any, bigram_name, k_values_to_plot, title, filename=None):
    plot_basics()

    # plot values
    for k_idx, k in enumerate(k_values_to_plot):
        if k_idx == 0:
            plt.plot(statistics_any[:, k_idx], label=f"Prediction Any Top {k}",
                     marker="^", markeredgecolor='none', color="g", linestyle='')
        elif k_idx < len(k_values_to_plot) - 1:
            plt.plot(statistics_any[:, k_idx], label=f"Prediction Any Top {k}",
                     marker="s", markeredgecolor='none', color="b", linestyle='')
        else:
            plt.plot(statistics_any[:, k_idx], label=f"Prediction Any Top {k}",
                     marker="v", markeredgecolor='none', color="r", linestyle='')

    plt.xlabel("Turn", fontsize=14)
    plt.xticks(np.arange(10), [str(i) for i in range(1, 11)])

    plt.ylabel("Prediction Accuracy", fontsize=14)
    plt.ylim((0, 1))
    plt.yticks(np.arange(0, 1.01, 0.1), ["{:0.1f}".format(i) for i in np.arange(0, 1.01, 0.1)])

    legend = plt.legend()
    frame = legend.get_frame()
    frame.set_facecolor('w')

    # plt.title(title, fontsize=16, pad=5)
    if filename:
        plt.savefig(f'{result_folder}/{filename}.png')
    plt.show()


def load_cluster_centroids():
    from analysis_tools import load_data_set, normalize_labels

    start_time = time.time()
    played_decks = load_data_set("ALL", True, "deck_data/Decks.json")
    end_time = time.time()
    logging.info("loading the deck data sets: {} s".format(end_time - start_time))

    import pickle
    with open("results_clustering\\labels.txt", "rb") as file:
        cluster_labels = pickle.load(file)
    best_clustering = {"single": 120, "complete": 90, "kmeans": 50, "dbscan-2": 59, "dbscan-5": 30, "dbscan-10": 30}

    from clustering_algorithms.fuzzy_deck_cluster import FuzzyDeckCluster
    cluster_data = {}
    for alg in best_clustering:
        labelled_decks = list(zip(played_decks, cluster_labels[alg][best_clustering[alg]]))
        centroids = []
        for i in set(cluster_labels[alg][best_clustering[alg]]):
            centroids.append(FuzzyDeckCluster([deck[0] for deck in labelled_decks if deck[1] == i]).centroid())

        cluster_data[alg] = {"centroids": centroids}
    return cluster_data


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    result_folder = "results_year_2019"

    # region load data
    # load validation data
    validation_files = ['replay_data/2019-{:02d}.json'.format(i) for i in {2,3,4}]

    minimal_validation_game_length = 15
    validation_data = load_dataset(validation_files, minimal_validation_game_length)
    logging.info(f"{len(validation_data)} validation games of minimum length {minimal_validation_game_length}")

    # load cluster datasets
    cluster_data = load_cluster_centroids()
    logging.info("cluster data sets loaded")
    # endregion

    # region plot top k statistic whole game
    # test prediction for whole game
    k = 10
    evaluate_turns = 10
    for clustering_name in cluster_data:
        cluster_centroids = cluster_data[clustering_name]["centroids"]

        statistics_player = np.zeros((evaluate_turns, k+1, 2))
        statistics_opponent = np.zeros((evaluate_turns, k+1, 2))

        for game in validation_data:
            cards_per_turn_player, cards_per_turn_opponent = get_turn_dicts(game)

            statistics_player += evaluate_top_k_predictions_turn_dict(cluster_centroids, cards_per_turn_player,
                                                                      evaluate_turns, k, only_next_turn=False)
            statistics_opponent += evaluate_top_k_predictions_turn_dict(cluster_centroids, cards_per_turn_opponent,
                                                                        evaluate_turns, k, only_next_turn=False)

        statistics = statistics_player + statistics_opponent

        # plot the highest ranked prediction accuracy
        cluster_data[clustering_name]["statistics"] = statistics

    """
        plot_statistics_per_top_k(statistics, clustering_name, k,
                                  title="card prediction per rank, remaining game",
                                  filename=f"per_rank_prediction_game_{clustering_name}")
    """
    # endregion

    """
    # region plot any k statistic whole game
    # plot summary statistics (is any of the k predictions correct#
    for clustering_name in bigram_data_sets:
        bigram_data = bigram_data_sets[clustering_name]["data"]
        k_values_to_plot = [1, 5, 10]

        statistics_any = np.zeros((evaluate_turns, len(k_values_to_plot)))

        statistics_player = np.zeros((evaluate_turns, k + 1, 2))
        statistics_opponent = np.zeros((evaluate_turns, k + 1, 2))

        for game in validation_data:
            cards_per_turn_player, cards_per_turn_opponent = get_turn_dicts(game)

            statistics_player += evaluate_top_k_predictions_turn_dict(bigram_data, cards_per_turn_player,
                                                                      evaluate_turns, k, calculate_any_k=True)
            statistics_opponent += evaluate_top_k_predictions_turn_dict(bigram_data, cards_per_turn_opponent,
                                                                        evaluate_turns, k, calculate_any_k=True)

        statistics = statistics_player + statistics_opponent

        for k_idx, k_val in enumerate(k_values_to_plot):
            statistics_any[:, k_idx] = statistics[:, k_val-1, 0] / (statistics[:, k_val-1, 0] + statistics[:, k_val-1, 1])

        plot_statistics_any_top_k(statistics_any, clustering_name, k_values_to_plot,
                                  title="aggregated card prediction, remaining game",
                                  filename=f"aggregated_prediction_game_{clustering_name}")
    # endregion


    # region plot top k statistic next turn
    # test prediction for whole game
    k = 10
    evaluate_turns = 10
    for clustering_name in bigram_data_sets:
        bigram_data = bigram_data_sets[clustering_name]["data"]

        statistics_player = np.zeros((evaluate_turns, k+1, 2))
        statistics_opponent = np.zeros((evaluate_turns, k+1, 2))

        for game in validation_data:
            cards_per_turn_player, cards_per_turn_opponent = get_turn_dicts(game)

            statistics_player += evaluate_top_k_predictions_turn_dict(bigram_data, cards_per_turn_player,
                                                                      evaluate_turns, k, only_next_turn=True)
            statistics_opponent += evaluate_top_k_predictions_turn_dict(bigram_data, cards_per_turn_opponent,
                                                                        evaluate_turns, k, only_next_turn=True)

        statistics = statistics_player + statistics_opponent

        # plot the highest ranked prediction accuracy
        bigram_data_sets[clustering_name]["statistics"] = statistics

        plot_statistics_per_top_k(statistics, clustering_name, k,
                                  title="card prediction per rank, next turn",
                                  filename=f"per_rank_prediction_turn_{clustering_name}")
    # endregion

    # region plot any k statistic whole game
    # plot summary statistics (is any of the k predictions correct
    for clustering_name in bigram_data_sets:
        bigram_data = bigram_data_sets[clustering_name]["data"]

        k_values_to_plot = [10, 5, 1]
        statistics_any = np.zeros((evaluate_turns, len(k_values_to_plot)))

        statistics_player = np.zeros((evaluate_turns, k + 1, 2))
        statistics_opponent = np.zeros((evaluate_turns, k + 1, 2))

        for game in validation_data:
            cards_per_turn_player, cards_per_turn_opponent = get_turn_dicts(game)

            statistics_player += evaluate_top_k_predictions_turn_dict(bigram_data, cards_per_turn_player,
                                                                      evaluate_turns, k, only_next_turn=True,
                                                                      calculate_any_k=True)
            statistics_opponent += evaluate_top_k_predictions_turn_dict(bigram_data, cards_per_turn_opponent,
                                                                        evaluate_turns, k, only_next_turn=True,
                                                                        calculate_any_k=True)

        statistics = statistics_player + statistics_opponent

        for k_idx, k_val in enumerate(k_values_to_plot):
            statistics_any[:, k_idx] = statistics[:, k_val-1, 0] / (statistics[:, k_val-1, 0] + statistics[:, k_val-1, 1])

        plot_statistics_any_top_k(statistics_any, clustering_name, k_values_to_plot,
                                  title="aggregrated card prediction, next turn",
                                  filename=f"aggregated_prediction_turn_{clustering_name}")
    # endregion
    """