import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable


def heatmap(data, row_labels=None, col_labels=None, ax=None, cbar=True,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    if cbar:
        # Create colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="30%", pad=0.10)

        cbar = ax.figure.colorbar(im, cax=cax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    if row_labels and col_labels:
        # We want to show all ticks...
        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_yticks(np.arange(data.shape[0]))
        # ... and label them with the respective list entries.
        ax.set_xticklabels(col_labels, fontsize=12)
        ax.set_yticklabels(row_labels, fontsize=12)

        # Let the horizontal axes labeling appear on top.
        ax.tick_params(top=True, bottom=False,
                       labeltop=True, labelbottom=False)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        # ... and label them with the respective list entries.
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # Let the horizontal axes labeling appear on top.
        ax.tick_params(top=False, bottom=False,
                       labeltop=False, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=4)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, ignore_diagonal=False, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if i == j and ignore_diagonal:
                continue
            kw.update(color=textcolors[int(im.norm(data[i, j]) < threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), fontsize=10, **kw)
            texts.append(text)

    return texts


if __name__ == "__main__":
    bot_names = ["Proposed Agent", "Tyche (MCTS)", "Tyche (One-step Lookahead)", "Alpha Beta Pruning", "Greedy Agent 1",
                 "Pruned BFS", "Álvaro (MCTS)", "Beam Search", "Greedy Agent 2"]
    deck_names = ["Aggro Pirate Warrior", "Midrange Jade Shaman", "Reno Kazakus Mage",
                  "Midrange Buff Paladin", "Miracle Pirate Rogue", "Zoo Discard Warlock"]

    win_rates = np.zeros((9, 9))
    detailed_game_outcomes = np.zeros((9, 9, 6, 6))
    with open("DissertationResults.txt", "r") as f:
        for line in f.readlines():
            line = line.split(" ")
            #if line[0].startswith("Bot"):
            #    bot_names.append(line[-1][:-1])
            if line[0].startswith("MasterMatchResult:"):
                i, j, total_wins_i, total_wins_j = [int(i) for i in line[1:5]]
                if i == 9 or j == 9:
                    continue
                win_rates[i, j] = total_wins_i / (total_wins_i + total_wins_j)
                win_rates[j, i] = total_wins_j / (total_wins_i + total_wins_j)

                bot_i_wins = np.array([int(i) for i in line[5:41]]).reshape((6, 6))
                bot_j_wins = np.array([int(i) for i in line[41:77]]).reshape((6, 6))

                detailed_game_outcomes[i, j, :, :] = bot_i_wins / (bot_i_wins + bot_j_wins)
                detailed_game_outcomes[j, i, :, :] = bot_j_wins / (bot_i_wins + bot_j_wins)
                detailed_game_outcomes[j, i, :, :] = detailed_game_outcomes[j, i, :, :].transpose()


    import matplotlib.font_manager as font_manager
    plt.rc('pgf', texsystem='lualatex')
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{kmath}\usepackage{kerkis}\renewcommand\sfdefault\rmdefault')
    font_path = 'c://windows//fonts//kerkis.ttf'
    prop = font_manager.FontProperties(fname=font_path)


    # plot win-rate filtered by known decks
    average_win_rates = [0]*9
    for i in range(len(win_rates)):
        average_win_rates[i] = np.mean(np.append(win_rates[i, :i], win_rates[i, (i + 1):]))

    grid = dict(height_ratios=[win_rates.shape[0]], width_ratios=[win_rates.shape[0], 1])
    fig, axes = plt.subplots(ncols=2, nrows=1, gridspec_kw = grid)
    fig.set_figheight(8)
    fig.set_figwidth(8)

    im, cbar = heatmap(np.array(average_win_rates).reshape(9, 1), None, None, ax=axes[1], vmin=0, vmax=1, cbar=False)
    im.axes.set_ylabel("average win-rate", fontsize=12)

    texts = annotate_heatmap(im, valfmt="{x:.2f}", threshold=0.5)

    im, _ = heatmap(win_rates, bot_names[:9], bot_names[0:9], ax=axes[0], vmin=0, vmax=1, cbar=False)
    _ = annotate_heatmap(im, valfmt="{x:.2f}", threshold=0.6, ignore_diagonal=True, fontproperties=prop)

    plt.subplots_adjust(top=0.95, bottom=-0.25, left=0.20, right=0.92, hspace=0.0,
                        wspace=0.0)

    fig.tight_layout()
    plt.savefig("average_results.pdf")
    plt.show()


    # plot win-rate filtered by known decks
    filtered_win_rates = np.zeros((9, 9))
    for i in range(len(win_rates)):
        for j in range(len(win_rates)):
            filtered_win_rates[i, j] = np.mean(detailed_game_outcomes[i, j, :4, :4])

    total_win_rate = [0]*9
    for i in range(len(win_rates)):
        total_win_rate[i] = np.mean(np.append(filtered_win_rates[i, :i], filtered_win_rates[i, (i + 1):]))

    grid = dict(height_ratios=[filtered_win_rates.shape[0]], width_ratios=[filtered_win_rates.shape[0], 1])
    fig, axes = plt.subplots(ncols=2, nrows=1, gridspec_kw = grid)
    fig.set_figheight(8)
    fig.set_figwidth(8)

    im, cbar = heatmap(np.array(total_win_rate).reshape(9, 1), None, None, ax=axes[1], vmin=0, vmax=1,
                       aspect="equal", cbar=False)
    im.axes.set_ylabel("average win-rate", fontsize=12)

    texts = annotate_heatmap(im, valfmt="{x:.2f}", threshold=0.6)

    im, _ = heatmap(filtered_win_rates, bot_names[:9], bot_names[0:9], ax=axes[0], vmin=0, vmax=1,
                       aspect="equal", cbar=False)
    _ = annotate_heatmap(im, valfmt="{x:.2f}", threshold=0.5, ignore_diagonal=True, fontproperties=prop)

    plt.subplots_adjust(top=0.95, bottom=-0.25, left=0.20, right=0.95, hspace=0.0,
                        wspace=0.0)

    fig.tight_layout()
    plt.savefig("filtered_average_results.pdf")
    plt.show()


    # plotting win-rates against a single opponent
    for i in range(1,9):
        result = detailed_game_outcomes[0, i, :, :]

        win_rate_using_deck = np.mean(result, axis=1)
        win_rate_against_deck = np.mean(result, axis=0)

        grid = dict(height_ratios=[result.shape[0], 1], width_ratios=[result.shape[0], 1])
        fig, axes = plt.subplots(ncols=2, nrows=2, gridspec_kw = grid)
        fig.set_figheight(6)
        fig.set_figwidth(6)

        im, _ = heatmap(np.array(win_rate_using_deck).reshape(6, 1), None, None, ax=axes[0, 1], vmin=0, vmax=1,
                           aspect="equal", cbar=False)
        im.axes.set_ylabel("average win-rate playing a deck", fontsize=12)

        _ = annotate_heatmap(im, valfmt="{x:.2f}", threshold=0.6)

        im, _ = heatmap(np.array(win_rate_against_deck).reshape(1, 6), None, None, ax=axes[1, 0], vmin=0, vmax=1,
                           aspect="equal", cbar=False)
        im.axes.set_xlabel("average win-rate playing against a deck", fontsize=12)
        im.axes.xaxis.set_label_position("top")
        #pos = im.axes._position
        #im.axes.set_position([pos.x0, pos.x1, pos.width, pos.height])

        _ = annotate_heatmap(im, valfmt="{x:.2f}", threshold=0.6)

        im, _ = heatmap(result, deck_names, deck_names, ax=axes[0, 0], vmin=0, vmax=1,
                        aspect="equal", cbar=False)
        _ = annotate_heatmap(im, valfmt="{x:.1f}", threshold=0.5, ignore_diagonal=False, fontproperties=prop)

        axes[1, 1].axis("off")

        plt.subplots_adjust(top=0.95, bottom=-0.25, left=0.20, right=0.95, hspace=0.0,
                            wspace=0.0)
        plt.axis('off')
        fig.tight_layout()
        plt.savefig(f"result_opponent_{i}.pdf")

        plt.show()


    # plotting win-rates per deck against all opponents
    result = np.mean(detailed_game_outcomes[0, 1:, :, :], axis=0)

    win_rate_using_deck = np.mean(result, axis=1)
    win_rate_against_deck = np.mean(result, axis=0)

    grid = dict(height_ratios=[result.shape[0], 1], width_ratios=[result.shape[0], 1])
    fig, axes = plt.subplots(ncols=2, nrows=2, gridspec_kw = grid)
    fig.set_figheight(6)
    fig.set_figwidth(6)

    im, _ = heatmap(np.array(win_rate_using_deck).reshape(6, 1), None, None, ax=axes[0, 1], vmin=0, vmax=1,
                       aspect="equal", cbar=False)
    im.axes.set_ylabel("average win-rate playing a deck", fontsize=12)

    _ = annotate_heatmap(im, valfmt="{x:.2f}", threshold=0.6)

    im, _ = heatmap(np.array(win_rate_against_deck).reshape(1, 6), None, None, ax=axes[1, 0], vmin=0, vmax=1,
                       aspect="equal", cbar=False)
    im.axes.set_xlabel("average win-rate playing against a deck", fontsize=12)
    im.axes.xaxis.set_label_position("top")
    #pos = im.axes._position
    #im.axes.set_position([pos.x0, pos.x1, pos.width, pos.height])

    _ = annotate_heatmap(im, valfmt="{x:.2f}", threshold=0.6)

    im, _ = heatmap(result, deck_names, deck_names, ax=axes[0, 0], vmin=0, vmax=1,
                    aspect="equal", cbar=False)
    _ = annotate_heatmap(im, valfmt="{x:.2f}", threshold=0.5, ignore_diagonal=False, fontproperties=prop)

    axes[1, 1].axis("off")

    plt.subplots_adjust(top=0.95, bottom=-0.25, left=0.20, right=0.95, hspace=0.0,
                        wspace=0.0)
    plt.axis('off')
    fig.tight_layout()
    plt.savefig(f"result_per_deck_proposed_vs_all.pdf")
    plt.show()


    # plotting average win-rates per deck over all matchups
    result = np.zeros((6, 6))
    times = 0
    for i in range(0, 9):
        for j in range(0, 9):
            if i == j:
                continue
            result += detailed_game_outcomes[i, j, :, :]
            times += 1
    result = result/times

    win_rate_using_deck = np.mean(result, axis=1)
    win_rate_against_deck = np.mean(result, axis=0)

    grid = dict(height_ratios=[result.shape[0], 1], width_ratios=[result.shape[0], 1])
    fig, axes = plt.subplots(ncols=2, nrows=2, gridspec_kw = grid)
    fig.set_figheight(6)
    fig.set_figwidth(6)

    im, _ = heatmap(np.array(win_rate_using_deck).reshape(6, 1), None, None, ax=axes[0, 1], vmin=0, vmax=1,
                       aspect="equal", cbar=False)
    im.axes.set_ylabel("average win-rate playing a deck", fontsize=12)

    _ = annotate_heatmap(im, valfmt="{x:.2f}", threshold=0.5)

    im, _ = heatmap(np.array(win_rate_against_deck).reshape(1, 6), None, None, ax=axes[1, 0], vmin=0, vmax=1,
                       aspect="equal", cbar=False)
    im.axes.set_xlabel("average win-rate playing against a deck", fontsize=12)
    im.axes.xaxis.set_label_position("top")

    _ = annotate_heatmap(im, valfmt="{x:.2f}", threshold=0.5)

    im, _ = heatmap(result, deck_names, deck_names, ax=axes[0, 0], vmin=0, vmax=1,
                    aspect="equal", cbar=False)
    _ = annotate_heatmap(im, valfmt="{x:.2f}", threshold=0.5, ignore_diagonal=False, fontproperties=prop)

    axes[1, 1].axis("off")

    plt.subplots_adjust(top=0.95, bottom=-0.25, left=0.20, right=0.95, hspace=0.0,
                        wspace=0.0)
    plt.axis('off')
    fig.tight_layout()
    plt.savefig(f"result_per_deck_all_vs_all.pdf")
    plt.show()
