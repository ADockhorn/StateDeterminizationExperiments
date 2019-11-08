from itertools import combinations, product


def add_isolated_turns(bigram_data, turn_dict):
    if not turn_dict:
        return

    for i in range(max(turn_dict)+1):
        if i in turn_dict:
            for card in turn_dict[i]:
                if card not in bigram_data:
                    bigram_data[card] = {}
            for bigram in combinations(turn_dict[i], 2):
                if bigram[1] in bigram_data[bigram[0]]:
                    bigram_data[bigram[0]][bigram[1]] += 1
                else:
                    bigram_data[bigram[0]][bigram[1]] = 1

                if bigram[0] in bigram_data[bigram[1]]:
                    bigram_data[bigram[1]][bigram[0]] += 1
                else:
                    bigram_data[bigram[1]][bigram[0]] = 1
    return bigram_data


def add_succeeding_turns(bigram_data, turn_dict):
    if not turn_dict:
        return bigram_data

    for i in range(max(turn_dict)):
        if i in turn_dict and i + 1 in turn_dict:
            for card in turn_dict[i]:
                if card not in bigram_data:
                    bigram_data[card] = {}
            for card in turn_dict[i+1]:
                if card not in bigram_data:
                    bigram_data[card] = {}

            for bigram in product(turn_dict[i], turn_dict[i + 1]):
                if bigram[1] in bigram_data[bigram[0]]:
                    bigram_data[bigram[0]][bigram[1]] += 1
                else:
                    bigram_data[bigram[0]][bigram[1]] = 1
    return bigram_data


def add_combined_turns(bigram_data, turn_dict):
    add_succeeding_turns(bigram_data, turn_dict)
    add_isolated_turns(bigram_data, turn_dict)
    return bigram_data


def add_whole_game(bigram_data, turn_dict):
    if not turn_dict:
        return bigram_data

    all_turns = []
    for i in turn_dict:
        all_turns.extend(turn_dict[i])

    for card in all_turns:
        if card not in bigram_data:
            bigram_data[card] = {}

    for bigram in combinations(all_turns, 2):
        if bigram[1] in bigram_data[bigram[0]]:
            bigram_data[bigram[0]][bigram[1]] += 1
        else:
            bigram_data[bigram[0]][bigram[1]] = 1

        if bigram[0] in bigram_data[bigram[1]]:
            bigram_data[bigram[1]][bigram[0]] += 1
        else:
            bigram_data[bigram[1]][bigram[0]] = 1

    return bigram_data


def add_whole_game_sequence(bigram_data, turn_dict):
    if not turn_dict:
        return bigram_data

    all_turns = []
    for i in range(max(turn_dict)+1):
        if i in turn_dict:
            all_turns.extend(turn_dict[i])

    for card in all_turns:
        if card not in bigram_data:
            bigram_data[card] = {}

    for bigram in combinations(all_turns, 2):
        if bigram[1] in bigram_data[bigram[0]]:
            bigram_data[bigram[0]][bigram[1]] += 1
        else:
            bigram_data[bigram[0]][bigram[1]] = 1

    return bigram_data


def train_bigram_data(training_data):
    bigram_data_whole_game_sequence = dict()
    bigram_data_whole_game = dict()
    bigram_data_combined = dict()
    bigram_data_succeeding = dict()
    bigram_data_isolated = dict()

    training_games = len(training_data)
    for game_id, game in enumerate(training_data):
        print(f"processing game nr {game_id+1} out of {training_games}")

        cards_per_turn_player = dict()
        cards_per_turn_opponent = dict()
        for action in game["card_history"]:
            target_dict = cards_per_turn_player if action["player"] == "me" else cards_per_turn_opponent
            if action["turn"] in target_dict:
                target_dict[action["turn"]].append(action["card"]["id"])
            else:
                target_dict[action["turn"]] = [action["card"]["id"]]

        add_isolated_turns(bigram_data_isolated, cards_per_turn_opponent)
        add_isolated_turns(bigram_data_isolated, cards_per_turn_player)

        add_succeeding_turns(bigram_data_succeeding, cards_per_turn_opponent)
        add_succeeding_turns(bigram_data_succeeding, cards_per_turn_player)

        add_combined_turns(bigram_data_combined, cards_per_turn_opponent)
        add_combined_turns(bigram_data_combined, cards_per_turn_player)

        add_whole_game(bigram_data_whole_game, cards_per_turn_opponent)
        add_whole_game(bigram_data_whole_game, cards_per_turn_player)

        add_whole_game_sequence(bigram_data_whole_game_sequence, cards_per_turn_opponent)
        add_whole_game_sequence(bigram_data_whole_game_sequence, cards_per_turn_player)

    return {"isolated": {"data": bigram_data_isolated},
            "succeeding": {"data": bigram_data_succeeding},
            "combined": {"data": bigram_data_combined},
            "game": {"data": bigram_data_whole_game},
            "game_sequence": {"data": bigram_data_whole_game_sequence}}


if __name__ == "__main__":
    turn_dict = {1: ["a"], 2: ["b", "c"], 3: ["d"]}

    bigram_data_whole_game_sequence = add_whole_game_sequence(dict(), turn_dict)
    bigram_data_whole_game = add_whole_game(dict(), turn_dict)
    bigram_data_combined = add_combined_turns(dict(), turn_dict)
    bigram_data_succeeding = add_succeeding_turns(dict(), turn_dict)
    bigram_data_isolated = add_isolated_turns(dict(), turn_dict)

    print("isolated", bigram_data_isolated)
    print("succeeding", bigram_data_succeeding)
    print("combined", bigram_data_combined)
    print("whole game", bigram_data_whole_game)
    print("whole game sequence", bigram_data_whole_game_sequence)
