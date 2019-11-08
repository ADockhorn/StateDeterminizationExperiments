import json
import numpy as np
import matplotlib.pyplot as plt
from bigram_extraction import *
import logging


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


def combine_two_predictions(base_prediction, new_prediction):
    # combine the two predictions
    for card, card_count in new_prediction.items():
        if card in base_prediction:
            base_prediction[card] += card_count
        else:
            base_prediction[card] = card_count


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


def evaluate_top_k_predictions_turn_dict(bigram_data, cards_per_turn, evaluate_turns, k, filter=False,
                                         only_next_turn=False, calculate_any_k=False):
    prediction = dict()
    statistics = np.zeros((evaluate_turns, k+1, 2))
    observed_cards = set()

    if not cards_per_turn:
        return statistics

    for turn in range(1, min(max(cards_per_turn)-1, evaluate_turns+1)):

        if turn in cards_per_turn:
            for card in cards_per_turn[turn]:
                observed_cards.add(card)
                if card in bigram_data:
                    new_prediction = {x: y for x, y in bigram_data[card].items()}
                    combine_two_predictions(prediction, new_prediction)

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


def plot_statistics_per_top_k(statistics, bigram_name, k, result_folder, filename=None, legendloc="best"):
    plot_basics()

    plt.plot(np.mean(np.array([statistics[:, i, 0] / (statistics[:, i, 0] + statistics[:, i, 1]) for i in range(0, k)]),
                     axis=0), label=f"Average Rank 1 - {k}", linestyle="dashed", alpha=0.5, markersize=7,linewidth=3)

    plt.plot(statistics[:, 0, 0] / (statistics[:, 0, 0] + statistics[:, 0, 1]), label="Rank 1",
             marker="^", markeredgecolor='none', color="g", linestyle='', markersize=7)

    """
    for i in range(1, k-2):
        plt.plot(statistics[:, i, 0] / (statistics[:, i, 0] + statistics[:, i, 1]),
                 marker="s", markeredgecolor='none', color="b", linestyle='')

    plt.plot(statistics[:, k-2, 0] / (statistics[:, k-2, 0] + statistics[:, k-2, 1]),
             label=f"Prediction ranked from 2 to {k-1}",
             marker="s", markeredgecolor='none', color="b", linestyle='')
    """

    plt.plot(statistics[:, k - 1, 0] / (statistics[:, k - 1, 0] + statistics[:, k - 1, 1]), label=f"Rank {k}",
             marker="v", markeredgecolor='none', color="r", linestyle='', markersize=7)

    plt.xlabel("Turn", fontsize=14)
    plt.xticks(np.arange(10), [str(i) for i in range(1, 11)])

    plt.ylabel("Prediction Accuracy", fontsize=14)
    plt.ylim((0, 1))
    plt.yticks(np.arange(0, 1.01, 0.25), ["{:0.2f}".format(i) for i in np.arange(0, 1.01, 0.25)])

    handles, labels = plt.gca().get_legend_handles_labels()
    order = [1, 0, 2]
    legend = plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], fontsize=14, loc=legendloc)
    frame = legend.get_frame()
    frame.set_facecolor('w')

    if filename:
        plt.savefig(f'{result_folder}/{filename}.png')
    plt.show()


def plot_statistics_any_top_k(statistics_any, baseline, k_values_to_plot, result_folder, filename=None, legendloc="best", legendcol=1):
    plot_basics()

    from matplotlib.cm import get_cmap
    cmap = [get_cmap("tab20b").colors[12], get_cmap("tab20b").colors[13],
            get_cmap("tab20b").colors[14], get_cmap("tab20b").colors[15]]

    # plot values
    for k_idx, k in enumerate(k_values_to_plot):
        if k_idx == 0:
            plt.plot(statistics_any[:, k_idx], label=f"Any Top {k}",
                     marker="D", markeredgecolor='none', linestyle='dashed', markersize=7, c=cmap[3], linewidth=4)
        elif k_idx < len(k_values_to_plot) - 1:
            plt.plot(statistics_any[:, k_idx], label=f"Any Top {k}",
                     marker="s", markeredgecolor='none', linestyle='dashed', markersize=7, c=cmap[2])
        else:
            plt.plot(statistics_any[:, k_idx], label=f"Any Top {k}",
                     marker="o", markeredgecolor='none', linestyle='dashed', markersize=7, c=cmap[1])

    plt.plot(baseline[:, 0, 0] / (baseline[:, 0, 0] + baseline[:, 0, 1]), label="Rank 1",
             marker="^", markeredgecolor='none', color=cmap[0], markersize=7, linestyle="solid", linewidth=2)

    plt.xlabel("Turn", fontsize=14)
    plt.xticks(np.arange(10), [str(i) for i in range(1, 11)])

    plt.ylabel("Prediction Accuracy", fontsize=14)
    plt.ylim((0, 1))
    plt.yticks(np.arange(0, 1.01, 0.25), ["{:0.2f}".format(i) for i in np.arange(0, 1.01, 0.25)])

    legend = plt.legend(fontsize=14, loc=legendloc, ncol=legendcol)
    frame = legend.get_frame()
    frame.set_facecolor('w')

    # plt.title(title, fontsize=16, pad=5)
    if filename:
        plt.savefig(f'{result_folder}/{filename}.png')
    plt.show()


if __name__ == "__main__":
    import pickle
    from tqdm import tqdm
    result_folder = "results_year_2019_bigrams"
    load_results = True
    logging.basicConfig(level=logging.INFO)


    # region load data
    if not load_results:

        # load training data
        # August - September 2019 (3282 games of length 10, and 808 of length 15
        # training_files = ['replay_data/2019-08-{:02d}.json'.format(i) for i in range(6, 32)] + \
        #                 ['replay_data/2019-09-{:02d}.json'.format(i) for i in range(1, 31)]

        # January-October 2017 (89399 of length 10 and 24989 of length 14)
        # training_files = ['replay_data/2017-{:02d}.json'.format(i) for i in range(1, 11)]
        training_files = ['replay_data/2019-{:02d}.json'.format(i) for i in {2, 3, 4}]

        minimal_training_game_length = 15
        training_data = load_dataset(training_files, minimal_training_game_length)
        print(f"{len(training_data)} training games of minimum length {minimal_training_game_length}")

        # extract bigram data
        bigram_data_sets = train_bigram_data(training_data)
        # endregion
    else:
        bigram_data_sets = pickle.load(open(f"{result_folder}\\bigram_based_predictors.txt", "rb"))

    # load validation data
    # validation_files = ['replay_data/2019-10-{:02d}.json'.format(i) for i in range(1, 8)]

    # November-December 2017 (10328 of length 10 and 2152 of length 15)
    # validation_files = ['replay_data/2017-{:02d}.json'.format(i) for i in range(11, 13)]
    validation_files = ['replay_data/2019-{:02d}.json'.format(i) for i in {2, 3, 4}]

    minimal_validation_game_length = 15
    validation_data = load_dataset(validation_files, minimal_validation_game_length)
    print(f"{len(validation_data)} validation games of minimum length {minimal_validation_game_length}")



    # region plot top k statistic whole game
    # test prediction for whole game
    k = 10
    evaluate_turns = 10
    k_values_to_plot = [10, 5, 2]

    for bigram_name in bigram_data_sets:
        bigram_data = bigram_data_sets[bigram_name]["data"]
        logging.info(bigram_name)

        if "statistics" in bigram_data_sets[bigram_name]:
            continue

        statistics_player = np.zeros((evaluate_turns, k+1, 2))
        statistics_opponent = np.zeros((evaluate_turns, k+1, 2))

        for game in tqdm(validation_data):
            cards_per_turn_player, cards_per_turn_opponent = get_turn_dicts(game)

            statistics_player += evaluate_top_k_predictions_turn_dict(bigram_data, cards_per_turn_player, 
                                                                      evaluate_turns, k, only_next_turn=False)
            statistics_opponent += evaluate_top_k_predictions_turn_dict(bigram_data, cards_per_turn_opponent, 
                                                                        evaluate_turns, k, only_next_turn=False)

        statistics = statistics_player + statistics_opponent

        # plot the highest ranked prediction accuracy
        bigram_data_sets[bigram_name]["statistics"] = statistics

    # endregion

    # region plot any k statistic whole game
    # plot summary statistics (is any of the k predictions correct#
    for bigram_name in bigram_data_sets:
        bigram_data = bigram_data_sets[bigram_name]["data"]
        logging.info(bigram_name)

        if "statistics_any_game" in bigram_data_sets[bigram_name]:
            continue

        statistics_any = np.zeros((evaluate_turns, len(k_values_to_plot)))

        statistics_player = np.zeros((evaluate_turns, k + 1, 2))
        statistics_opponent = np.zeros((evaluate_turns, k + 1, 2))

        for game in tqdm(validation_data):
            cards_per_turn_player, cards_per_turn_opponent = get_turn_dicts(game)

            statistics_player += evaluate_top_k_predictions_turn_dict(bigram_data, cards_per_turn_player,
                                                                      evaluate_turns, k, calculate_any_k=True)
            statistics_opponent += evaluate_top_k_predictions_turn_dict(bigram_data, cards_per_turn_opponent,
                                                                        evaluate_turns, k, calculate_any_k=True)

        statistics = statistics_player + statistics_opponent

        for k_idx, k_val in enumerate(k_values_to_plot):
            statistics_any[:, k_idx] = statistics[:, k_val-1, 0] / (statistics[:, k_val-1, 0] + statistics[:, k_val-1, 1])

        bigram_data_sets[bigram_name]["statistics_any_game"] = statistics_any
    # endregion


    # region plot top k statistic next turn
    # test prediction for whole game
    k = 10
    evaluate_turns = 10
    for bigram_name in bigram_data_sets:
        bigram_data = bigram_data_sets[bigram_name]["data"]
        logging.info(bigram_name)

        if "statistics_next_turn" in bigram_data_sets[bigram_name]:
            continue

        statistics_player = np.zeros((evaluate_turns, k+1, 2))
        statistics_opponent = np.zeros((evaluate_turns, k+1, 2))

        for game in tqdm(validation_data):
            cards_per_turn_player, cards_per_turn_opponent = get_turn_dicts(game)

            statistics_player += evaluate_top_k_predictions_turn_dict(bigram_data, cards_per_turn_player,
                                                                      evaluate_turns, k, only_next_turn=True)
            statistics_opponent += evaluate_top_k_predictions_turn_dict(bigram_data, cards_per_turn_opponent,
                                                                        evaluate_turns, k, only_next_turn=True)

        statistics = statistics_player + statistics_opponent

        # plot the highest ranked prediction accuracy
        bigram_data_sets[bigram_name]["statistics_next_turn"] = statistics


    # endregion

    # region plot any k statistic whole game
    # plot summary statistics (is any of the k predictions correct
    for bigram_name in bigram_data_sets:
        bigram_data = bigram_data_sets[bigram_name]["data"]
        logging.info(bigram_name)

        if "statistics_any_next_turn" in bigram_data_sets[bigram_name]:
            continue

        statistics_any = np.zeros((evaluate_turns, len(k_values_to_plot)))

        statistics_player = np.zeros((evaluate_turns, k + 1, 2))
        statistics_opponent = np.zeros((evaluate_turns, k + 1, 2))

        for game in tqdm(validation_data):
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
        bigram_data_sets[bigram_name]["statistics_any_next_turn"] = statistics_any


    # endregion

    with open(f"{result_folder}\\bigram_based_predictors.txt", "wb") as file:
        pickle.dump(bigram_data_sets, file)

    for alg in bigram_data_sets:
        plot_statistics_per_top_k(bigram_data_sets[alg]["statistics"], alg, k,
                                  result_folder=result_folder,
                                  filename=f"per_rank_prediction_game_{alg}", legendloc="lower left")

    for alg in bigram_data_sets:
        plot_statistics_any_top_k(bigram_data_sets[alg]["statistics_any_game"],
                                  bigram_data_sets[alg]["statistics"], k_values_to_plot,
                                  result_folder=result_folder,
                                  filename=f"aggregated_prediction_game_{alg}", legendloc="lower left", legendcol=2)

    for alg in bigram_data_sets:
        plot_statistics_per_top_k(bigram_data_sets[alg]["statistics_next_turn"], alg, k,
                                  result_folder=result_folder, legendloc="upper right",
                                 filename=f"per_rank_prediction_turn_{alg}")

    for alg in bigram_data_sets:
        plot_statistics_any_top_k(bigram_data_sets[alg]["statistics_any_next_turn"],
                                  bigram_data_sets[alg]["statistics_next_turn"], k_values_to_plot,
                                  result_folder=result_folder, legendloc="upper left", legendcol=2,
                                  filename=f"aggregated_prediction_turn_{alg}")
