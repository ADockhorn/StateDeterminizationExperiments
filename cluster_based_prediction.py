from clustering_algorithms.fuzzy_deck_cluster import FuzzyDeckCluster, FuzzyDeck
import numpy as np
from prediction_test_bigrams import evaluate_top_k_predictions_next_turn, evaluate_top_k_predictions_remaining_game, get_turn_dicts
import json
import time
import logging
from prediction_test_bigrams import plot_statistics_any_top_k, plot_statistics_per_top_k


def load_dataset(filenames, minimal_length):
    dataset = []
    for file in filenames:
        with open(file) as json_file:
            data = json.load(json_file)
            dataset.extend([game for game in data["games"]
                            if max([0] + [action["turn"] for action in game["card_history"]]) >= minimal_length])
    return dataset


class ClusterBasedPrediction:

    def __init__(self, deck_data, labels):
        tmp_centroids = []
        deck_data = np.array(deck_data)
        labels = np.array(labels)
        for i in set(labels):
            tmp_centroids.append(FuzzyDeckCluster(deck_data[i == labels]).centroid())
        self.centroids = self.remap_centroid(tmp_centroids)
        self.observed_cards = dict()
        self.turn_stats = dict()

    def remap_centroid(self, centroids):
        card_map = dict()
        with open("deck_data/Cards_en.json") as f:
            data = json.load(f)
            for card_entry in data:
                if "dbfId" in card_entry and "id" in card_entry:
                    card_map[card_entry["dbfId"]] = card_entry["id"]

        new_centroids = [{card_map[int(card)]: centroid.card_multiset[card] for card in centroid.card_multiset if int(card) in card_map} for centroid in centroids]
        new_centroids = [FuzzyDeck({"card_multiset": centroid}) for centroid in new_centroids]
        return new_centroids

    def reset_observed_cards(self):
        self.observed_cards = dict()

    def observe_card(self, card):
        if card in self.observed_cards:
            self.observed_cards[card].append(1.0)
        else:
            self.observed_cards[card] = [1.0]

    def predict_top_k(self, k, turn):
        fuzzy_deck = FuzzyDeck({"card_multiset": self.observed_cards})
        closest_centroid = (self.centroids[0], 1.0)
        for centroid in self.centroids:
            distance = fuzzy_deck.jaccard_distance(centroid)
            if distance < closest_centroid[1]:
                closest_centroid = (centroid, distance)

        if turn in self.turn_stats:
            self.turn_stats[turn][0] += closest_centroid[1]
            self.turn_stats[turn][1] += 1
        else:
            self.turn_stats[turn] = [closest_centroid[1], 1]

        return {card: sum(closest_centroid[0].card_multiset[card]) for card in closest_centroid[0].card_multiset}


def evaluate_top_k_predictions_turn_dict(predictor, cards_per_turn, evaluate_turns, k, filter=False,
                                         only_next_turn=False, calculate_any_k=False):
    prediction = dict()
    statistics = np.zeros((evaluate_turns, k+1, 2))
    observed_cards = set()

    if not cards_per_turn:
        return statistics

    predictor.reset_observed_cards()

    for turn in range(1, min(max(cards_per_turn)-1, evaluate_turns+1)):

        if turn in cards_per_turn:
            for card in cards_per_turn[turn]:
                observed_cards.add(card)
                predictor.observe_card(card)
            prediction = predictor.predict_top_k(k, turn)

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


def plot_turn_stats(stats, result_folder, filename):
    from prediction_test_bigrams import plot_basics
    import matplotlib.pyplot as plt

    plot_basics()
    for alg in stats:
        plt.plot(range(1, 11), stats[alg]["turn_distance_stats"], label=alg, marker="")
    plt.ylim((0.75, 1.00))
    plt.legend()
    # plt.savefig(f"{result_folder}\\{filename}.png")
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from tqdm import tqdm

    result_folder = "results_year_2019"
    load_clusters = True

    # region load data
    # load validation data
    validation_files = ['replay_data/2019-{:02d}.json'.format(i) for i in {2, 3, 4}]

    minimal_validation_game_length = 15
    validation_data = load_dataset(validation_files, minimal_validation_game_length)
    logging.info(f"{len(validation_data)} validation games of minimum length {minimal_validation_game_length}")

    import pickle
    if not load_clusters:
        # load cluster datasets
        cluster_data = load_cluster_centroids()
        logging.info("cluster data sets loaded")
        # endregion

        # region create Predictor
        from analysis_tools import load_data_set
        start_time = time.time()
        played_decks = load_data_set("ALL", True, "deck_data/Decks.json")
        end_time = time.time()
        logging.info("loading the deck data sets: {} s".format(end_time - start_time))

        with open("results_clustering\\labels.txt", "rb") as file:
            cluster_labels = pickle.load(file)
        best_clustering = {"single": 120, "complete": 90, "kmeans": 50, "dbscan-2": 59, "dbscan-5": 30, "dbscan-10": 30}

        cluster_based_predictors = dict()
        for alg in best_clustering:
            cluster_based_predictors[alg] = {"predictor": ClusterBasedPrediction(played_decks, cluster_labels[alg][best_clustering[alg]])}
        # endregion
    else:
        cluster_based_predictors = pickle.load(open(f"{result_folder}\\cluster_based_predictors.txt", "rb"))

    # region test prediction
    k = 10
    evaluate_turns = 10

    logging.info("cards remaining game")
    for alg in cluster_based_predictors:
        logging.info(alg)
        if "statistics" in cluster_based_predictors[alg]:
            continue

        predictor = cluster_based_predictors[alg]["predictor"]
        predictor.turn_stats = dict()

        statistics_player = np.zeros((evaluate_turns, k + 1, 2))
        statistics_opponent = np.zeros((evaluate_turns, k + 1, 2))

        for game in tqdm(validation_data):
            cards_per_turn_player, cards_per_turn_opponent = get_turn_dicts(game)

            statistics_player += evaluate_top_k_predictions_turn_dict(predictor, cards_per_turn_player,
                                                                      evaluate_turns, k, only_next_turn=False)
            statistics_opponent += evaluate_top_k_predictions_turn_dict(predictor, cards_per_turn_opponent,
                                                                        evaluate_turns, k, only_next_turn=False)

        statistics = statistics_player + statistics_opponent

        # plot the highest ranked prediction accuracy
        cluster_based_predictors[alg]["statistics"] = statistics
        cluster_based_predictors[alg]["turn_distance_stats"] = [predictor.turn_stats[i][0]/predictor.turn_stats[i][1] for i in range(1,11)]

    # endregion

    # region plot any k statistic whole game
    # plot summary statistics (is any of the k predictions correct#
    logging.info("aggregated cards remaining game")
    for alg in cluster_based_predictors:
        logging.info(alg)
        if "statistics_any_game" in cluster_based_predictors[alg]:
            continue

        predictor = cluster_based_predictors[alg]["predictor"]
        k_values_to_plot = [10, 5, 2]

        statistics_any = np.zeros((evaluate_turns, len(k_values_to_plot)))

        statistics_player = np.zeros((evaluate_turns, k + 1, 2))
        statistics_opponent = np.zeros((evaluate_turns, k + 1, 2))

        for game in tqdm(validation_data):
            cards_per_turn_player, cards_per_turn_opponent = get_turn_dicts(game)

            statistics_player += evaluate_top_k_predictions_turn_dict(predictor, cards_per_turn_player,
                                                                      evaluate_turns, k, calculate_any_k=True)
            statistics_opponent += evaluate_top_k_predictions_turn_dict(predictor, cards_per_turn_opponent,
                                                                        evaluate_turns, k, calculate_any_k=True)

        statistics = statistics_player + statistics_opponent

        for k_idx, k_val in enumerate(k_values_to_plot):
            statistics_any[:, k_idx] = statistics[:, k_val-1, 0] / (statistics[:, k_val-1, 0] + statistics[:, k_val-1, 1])

        cluster_based_predictors[alg]["statistics_any_game"] = statistics_any
    # endregion

    # region plot top k statistic next turn
    # test prediction for whole game
    k = 10
    evaluate_turns = 10
    logging.info("cards next turn")
    for alg in cluster_based_predictors:
        logging.info(alg)
        predictor = cluster_based_predictors[alg]["predictor"]
        if "statistics_next_turn" in cluster_based_predictors[alg]:
            continue

        statistics_player = np.zeros((evaluate_turns, k+1, 2))
        statistics_opponent = np.zeros((evaluate_turns, k+1, 2))

        for game in tqdm(validation_data):
            cards_per_turn_player, cards_per_turn_opponent = get_turn_dicts(game)

            statistics_player += evaluate_top_k_predictions_turn_dict(predictor, cards_per_turn_player,
                                                                      evaluate_turns, k, only_next_turn=True)
            statistics_opponent += evaluate_top_k_predictions_turn_dict(predictor, cards_per_turn_opponent,
                                                                        evaluate_turns, k, only_next_turn=True)

        statistics = statistics_player + statistics_opponent

        # plot the highest ranked prediction accuracy
        cluster_based_predictors[alg]["statistics_next_turn"] = statistics


    # endregion

    # region plot any k statistic whole game
    # plot summary statistics (is any of the k predictions correct
    logging.info("aggregated cards next turn")
    for alg in cluster_based_predictors:
        logging.info(alg)
        if "statistics_any_next_turn" in cluster_based_predictors[alg]:
            continue

        predictor = cluster_based_predictors[alg]["predictor"]

        k_values_to_plot = [10, 5, 2]
        statistics_any = np.zeros((evaluate_turns, len(k_values_to_plot)))

        statistics_player = np.zeros((evaluate_turns, k + 1, 2))
        statistics_opponent = np.zeros((evaluate_turns, k + 1, 2))

        for game in tqdm(validation_data):
            cards_per_turn_player, cards_per_turn_opponent = get_turn_dicts(game)

            statistics_player += evaluate_top_k_predictions_turn_dict(predictor, cards_per_turn_player,
                                                                      evaluate_turns, k, only_next_turn=True,
                                                                      calculate_any_k=True)
            statistics_opponent += evaluate_top_k_predictions_turn_dict(predictor, cards_per_turn_opponent,
                                                                        evaluate_turns, k, only_next_turn=True,
                                                                        calculate_any_k=True)

        statistics = statistics_player + statistics_opponent

        for k_idx, k_val in enumerate(k_values_to_plot):
            statistics_any[:, k_idx] = statistics[:, k_val-1, 0] / (statistics[:, k_val-1, 0] + statistics[:, k_val-1, 1])

        cluster_based_predictors[alg]["statistics_any_next_turn"] = statistics_any


    # endregion
    
    with open(f"{result_folder}\\cluster_based_predictors.txt", "wb") as file:
        pickle.dump(cluster_based_predictors, file)

    k_values_to_plot = [10, 5, 2]

    for alg in cluster_based_predictors:
        plot_statistics_per_top_k(cluster_based_predictors[alg]["statistics"], alg, k,
                                  result_folder=result_folder,
                                  filename=f"per_rank_prediction_game_{alg}")

    for alg in cluster_based_predictors:
        plot_statistics_any_top_k(cluster_based_predictors[alg]["statistics_any_game"],
                                  cluster_based_predictors[alg]["statistics"], k_values_to_plot,
                                  result_folder=result_folder,
                                  filename=f"aggregated_prediction_game_{alg}",
                                  legendloc="lower right", legendcol=2)

    for alg in cluster_based_predictors:
        plot_statistics_per_top_k(cluster_based_predictors[alg]["statistics_next_turn"], alg, k,
                                  result_folder=result_folder,
                                  filename=f"per_rank_prediction_turn_{alg}",
                                  legendloc="upper right")

    for alg in cluster_based_predictors:
        plot_statistics_any_top_k(cluster_based_predictors[alg]["statistics_any_next_turn"],
                                  cluster_based_predictors[alg]["statistics_next_turn"], k_values_to_plot,
                                  result_folder=result_folder,
                                  filename=f"aggregated_prediction_turn_{alg}", legendcol=2, legendloc="upper right")

    plot_turn_stats(cluster_based_predictors,
                    result_folder=result_folder,
                    filename=f"turn_stats_comparison")

