import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

atom_cnt_col_labels = ["C", "N", "O", "F"]
atom_cnt_row_labels = [
    "Orig",
    "MOLGAN",
    "GraphVAE",
    "DP-GGAN(No DP)",
    "DP-GVAE(No DP)",
    "Graph-PATE(No DP)",
    "DP-GGAN(ε=10)",
    "DP-GVAE(ε=10)",
    "Graph-PATE(ε=10)",
    "DP-GGAN(ε=5)",
    "DP-GVAE(ε=5)",
    "Graph-PATE(ε=5)",
    "DP-GGAN(ε=1)",
    "DP-GVAE(ε=1)",
    "Graph-PATE(ε=1)",
]
atom_cnt_data = np.array(
    [
        [6.3192, 1.4039, 1.0479, 0.0547],
        [6.68, 1.23, 0.92, 0.08],
        [6.56, 0.92, 0.88, 0.1],
        [6.86, 0.8137, 1.0758, 0.005],
        [6.0988, 0.77, 0.8227, 0.0075],
        [5.1462, 0.8491, 1, 0.005],
        [6.55, 1.2222, 1.2222, 0],
        [5.6749, 0.7192, 0.8182, 0.0062],
        [2.14, 0.86, 0.22, 0.626],
        [6.75, 0.9166, 1.1333, 0],
        [5.53, 0.7156, 0.8348, 0.0051],
        [0.98, 0.36, 0.86, 0],
        [6.93, 0.68, 1.56, 0],
        [5.51, 0.72, 0.71, 0.0046],
        [0.49, 0.57, 0.4, 0],
    ]
)

edge_cnt_col_labels = ["Single", "Double", "Triple"]
edge_cnt_row_labels = atom_cnt_row_labels
edge_cnt_data = np.array(
    [
        [8.0453, 1.0797, 0.2766],
        [7.48, 0.54, 0.1],
        [6.08, 0.68, 0.18],
        [6.68, 0.5714, 0.152],
        [5.9411, 0.5027, 0.0669],
        [6.3712, 0.5846, 0.1346],
        [7.5, 0.65, 0.03],
        [6.6815, 0.5135, 0.0362],
        [3.08, 0.62, 0.12],
        [8.45, 0.208, 0.166],
        [6.354, 0.4975, 0.068],
        [1.5, 0.6, 0],
        [7.18, 0.43, 0],
        [6.188, 0.504, 0.0697],
        [1.5, 0.6, 0],
    ]
)

ring_cnt_col_labels = ["Tri", "Quad", "Pent", "Hex"]
ring_cnt_row_labels = atom_cnt_row_labels
ring_cnt_data = np.array(
    [
        [0.4684, 0.4928, 0.4832, 0.1566],
        [0.45, 0.4123, 0.3923, 0.1234],
        [0.5068, 0.2568, 0.1824, 0.0687],
        [0.5213, 0.3003, 0.4103, 0.05],
        [0.462, 0.2343, 0.203, 0.0784],
        [0.5475, 0.267, 0.3897, 0.058],
        [0.47, 0.27, 0.18, 0.08],
        [0.4772, 0.2822, 0.1806, 0.0802],
        [0.15, 0.41, 0.06, 0.02],
        [0.416, 0.25, 0.28, 0.05],
        [0.4403, 0.2536, 0.1869, 0.0805],
        [0.012, 0.265, 0.125, 0],
        [0.5, 0, 0.125, 0.1875],
        [0.4489, 0.2587, 0.1858, 0.0763],
        [0.035, 0.01, 0.002, 0],
    ]
)


def plot_experiment(ax, indices, row_labels, col_labels, data):
    row_labels = [row_labels[i] for i in indices]
    data = data[indices]
    # row_labels = list(reversed([row_labels[i] for i in indices]))
    # data = np.flip(data[indices], axis=0)

    models = row_labels
    blocks = [(cl, data[:, i]) for i, cl in enumerate(col_labels)]

    bottom = np.zeros(len(row_labels))
    bar_width = 0.8

    # colors for each column
    colors = ['steelblue', 'orange', "yellowgreen", "tomato"]

    for i, (cl, weight) in enumerate(blocks):
        ax.bar(models, weight, bar_width, label=cl, bottom=bottom, color=colors[i])  # vertical bar
        # ax.barh(models, weight, bar_width, label=cl, left=bottom, color=colors[i])  # horizontal bar
        bottom += weight

    ax.legend(loc="upper center", ncol=len(col_labels), fontsize=8, bbox_to_anchor=(0.5, 1.15))


def plot_atom_cnt(ax, indices):
    plot_experiment(ax, indices, atom_cnt_row_labels, atom_cnt_col_labels, atom_cnt_data)
    ax.set_ylim(0, 10)


def plot_edge_cnt(ax, indices):
    plot_experiment(ax, indices, edge_cnt_row_labels, edge_cnt_col_labels, edge_cnt_data)
    ax.set_ylim(0, 10)


def plot_ring_cnt(ax, indices):
    plot_experiment(ax, indices, ring_cnt_row_labels, ring_cnt_col_labels, ring_cnt_data)
    ax.set_ylim(0, 2)


def plot_all(indices, name):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    plot_atom_cnt(ax1, indices)
    plot_edge_cnt(ax2, indices)
    plot_ring_cnt(ax3, indices)

    ax1.set_xlabel("atom count", fontsize=10, labelpad=10)
    ax2.set_xlabel("bond count", fontsize=10, labelpad=10)
    ax3.set_xlabel("ring count", fontsize=10, labelpad=10)

    ax1.tick_params(axis='x', rotation=90)
    ax2.tick_params(axis='x', rotation=90)
    ax3.tick_params(axis='x', rotation=90)

    # draw rectangles over Orig
    r1 = Rectangle((0.047, 0.28), 0.032, 0.589, alpha=0.2, facecolor='gold', clip_on=False)
    r2 = Rectangle((0.374, 0.28), 0.032, 0.589, alpha=0.2, facecolor='gold', clip_on=False)
    r3 = Rectangle((0.700, 0.28), 0.032, 0.589, alpha=0.2, facecolor='gold', clip_on=False)
    fig.add_artist(r1)
    fig.add_artist(r2)
    fig.add_artist(r3)

    # plt.subplots_adjust(left=0.3)
    fig.set_figwidth(10.6)
    fig.set_figheight(5)
    # fig.set_dpi(150)
    fig.tight_layout()

    plt.savefig(f'D:/Github/SEU-master-thesis/figures/{name}.pdf', pad_inches=-1.0)
    plt.show()


#
# modify indices to plot
#
plot_all([0, 1, 2, 3, 4, 5, 6, 7, 8], name='whole')  # whole
plot_all([0, 1, 2, 3, 4, 5, 6, 9, 12], name='dp_ggan')  # DP-GGAN
plot_all([0, 1, 2, 3, 4, 5, 7, 10, 13], name='dp_gvae')  # DP-GVAE
plot_all([0, 1, 2, 3, 4, 5, 8, 11, 14], name='graph_pate')  # Graph-PATE
