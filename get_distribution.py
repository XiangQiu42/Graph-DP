import numpy as np
import matplotlib.pyplot as plt
from util_dir.graph_utils import *
from utils import MolecularMetrics
from sklearn.neighbors import KernelDensity


def get_scores(all_smiles):
    logP = []
    qed = []
    sa = []

    for smiles in all_smiles:
        new_mol = Chem.MolFromSmiles(smiles)
        try:
            val_logP = Crippen.MolLogP(new_mol)
            val_qed = QED.qed(new_mol)
            val_sa = MolecularMetrics._compute_SAS(new_mol)

        except:
            continue
        logP.append(val_logP)
        qed.append(val_qed)
        sa.append(val_sa)

    return [qed, sa, logP]


def plot_hist(scores_orig, scores_gen, num_bins=42):
    for i in range(len(scores_orig)):
        plt.figure(i)
        # the original distribution
        plt.hist(scores_orig[i], bins=num_bins, density=True)

        # the generated distribution
        frequency_each, _, _ = plt.hist(scores_gen[i], num_bins, density=True,
                                        color='deepskyblue', alpha=0.7, histtype="bar")

        # plt.plot(frequency_each, color='palevioletred')

        plt.ylabel('normalized frequency', fontsize=14)
        if i == 0:
            plt.xlabel('QED', fontsize=14)
        elif i == 1:
            plt.xlabel('SA', fontsize=14)
        else:
            plt.xlabel('logP', fontsize=14)

        plt.show()


def plot_KDE_NoDP(true_scores, gen_scores):
    fig, ax = plt.subplots(1, 3)

    score_names = ['QED', 'SA', 'logP']

    colors = ["darkorange", "navy", "darkviolet", "darkolivegreen", 'red']
    methods = ["MolGAN", 'GraphVAE', "DP_GGAN", "DP-GVAE", "Graph-PATE"]

    for index, score_name in enumerate(score_names):
        if score_name == 'QED':
            X_plot = np.linspace(0.1, 0.7, 50)[:, np.newaxis]
            num_bins = np.linspace(0.1, 0.7, 50)
        elif score_name == 'SA':
            X_plot = np.linspace(-0.5, 8.5, 50)[:, np.newaxis]
            num_bins = np.linspace(-0.5, 8.5, 50)
        else:
            X_plot = np.linspace(-6, 7, 50)[:, np.newaxis]
            num_bins = np.linspace(-6, 7, 50)

        # the original distributions
        true_dens = np.array(true_scores[index])
        ax[index].hist(true_dens, num_bins, density=True, color='cornflowerblue', alpha=0.8, label='QM9', rwidth=0.8)
        # plt.hist(scores_gen[i], num_bins, density=True, color='deepskyblue', alpha=0.7, histtype="bar")

        _bandwidth = 0.016 if score_name == 'QED' else 0.2

        lw = 1.7

        for i, method in enumerate(methods):
            color = colors[i]
            X = np.array(gen_scores[i][index])[:, np.newaxis]

            kde = KernelDensity(kernel="gaussian", bandwidth=_bandwidth).fit(X)
            log_dens = kde.score_samples(X_plot)
            ax[index].plot(
                X_plot[:, 0],
                np.exp(log_dens),
                color=color,
                lw=lw,
                linestyle="-",
                label=method,
                # label="kernel = '{0}'".format(method),
            )

        # ax.text(6, 0.38, "N={0} points".format(N))

        ax[index].legend(loc="upper left", fontsize=12)
        ax[index].set_ylabel('normalized frequency', fontsize=12)
        ax[index].set_xlabel(score_name, fontsize=12)

        # ax.set_xlim(0, 1)
        # ax.set_ylim(-0.02, 0.4)

    fig.set_figwidth(16)
    fig.set_figheight(4.8)
    fig.tight_layout()
    plt.savefig(f'D:/Github/SEU-master-thesis/figures/nodp.pdf', pad_inches=0.0)
    plt.show()


def plot_KDE_eps(true_scores, gen_scores, name):
    fig, ax = plt.subplots(1, 3)

    score_names = ['QED', 'SA', 'logP']

    colors = ["darkorange", "navy", "darkviolet", "darkolivegreen", 'red']
    methods = ['No DP', r'$\varepsilon=10$', r"$\varepsilon=5$", r"$\varepsilon=1$"]

    for index, score_name in enumerate(score_names):
        if score_name == 'QED':
            X_plot = np.linspace(0.1, 0.7, 50)[:, np.newaxis]
            num_bins = np.linspace(0.1, 0.7, 50)
        elif score_name == 'SA':
            X_plot = np.linspace(-0.5, 8.5, 50)[:, np.newaxis]
            num_bins = np.linspace(-0.5, 8.5, 50)
        else:
            X_plot = np.linspace(-6, 7, 50)[:, np.newaxis]
            num_bins = np.linspace(-6, 7, 50)

        # the original distributions
        true_dens = np.array(true_scores[index])
        ax[index].hist(true_dens, num_bins, density=True, color='cornflowerblue', alpha=0.8, label='QM9', rwidth=0.8)
        # plt.hist(scores_gen[i], num_bins, density=True, color='deepskyblue', alpha=0.7, histtype="bar")

        _bandwidth = 0.016 if score_name == 'QED' else 0.2

        lw = 1.7

        for i, method in enumerate(methods):
            color = colors[i]
            X = np.array(gen_scores[i][index])[:, np.newaxis]

            if i == 1 and score_name == 'SA':
                X = np.array(gen_scores[3][index])[:, np.newaxis]

            if i == 2 and score_name == 'SA':
                X = np.array(gen_scores[1][index])[:, np.newaxis]

            if i == 3 and score_name == 'SA':
                X = np.array(gen_scores[2][index])[:, np.newaxis]

            if i > 0:
                _bandwidth = 0.05 if score_name == 'QED' else 0.5

            kde = KernelDensity(kernel="gaussian", bandwidth=_bandwidth).fit(X)
            log_dens = kde.score_samples(X_plot)
            ax[index].plot(
                X_plot[:, 0],
                np.exp(log_dens),
                color=color,
                lw=lw,
                linestyle="-",
                label=method,
                # label="kernel = '{0}'".format(method),
            )

        # ax.text(6, 0.38, "N={0} points".format(N))

        ax[index].legend(loc="upper left", fontsize=12)
        ax[index].set_ylabel('normalized frequency', fontsize=12)
        ax[index].set_xlabel(score_name, fontsize=12)

        # ax.set_xlim(0, 1)
        # ax.set_ylim(-0.02, 0.4)

    fig.set_figwidth(16)
    fig.set_figheight(4.8)
    fig.tight_layout()
    plt.savefig(f'D:/Github/SEU-master-thesis/figures/{name}.pdf', pad_inches=0.0)
    plt.show()


# data_root_5k = 'data/qm9_5k.smi'
# all_smiles_5k = load_smiles(data_root_5k)
# scores_5k = get_scores(all_smiles_5k)
# scores_5k = np.array(scores_5k)
# np.save('generated_mols/orig/qm9_5k', scores_5k)
#
# data_root_orig = 'data/smiles/smiles_qm9.pkl'
# all_smiles_orig = load_smiles(data_root_orig)
# score_orig = get_scores(all_smiles_orig)
#
# score_orig = np.array(score_orig)
# np.save('generated_mols/orig/qm9_whole', score_orig)
#
# plot_hist(score_orig, scores_5k)

# score_orig = np.load('generated_mols/orig/qm9_5k.npy')
score_orig = np.load('generated_mols/orig/qm9_whole.npy')

# pate_gen_root = 'generated_mols/gswgan_nodp_z32_pretrain_bs2048'
# gan_gen_root = 'generated_mols/GAN_nodp_z32_wgan_bs1024'
# vae_gen_root = 'generated_mols/vae_nodp_z32_bs2048'
# graphvae_root = 'generated_mols/generated_smiles'
# molgan_root = 'generated_mols/molgan'
#
# score_gan = get_scores(load_smiles(gan_gen_root))
# score_pate = get_scores(load_smiles(pate_gen_root))
# score_vae = get_scores(load_smiles(vae_gen_root))
# score_molgan = get_scores(load_smiles(molgan_root))
# score_graphvae = get_scores(load_smiles(graphvae_root))

root_nodp = 'generated_mols/GAN_nodp_z32_wgan_bs1024'
root_eps_10 = 'generated_mols/DPSGD_eps10_z32_7.07'
root_eps_5 = 'generated_mols/DPSGD_eps5_z32_3.97'
root_eps_1 = "generated_mols/DPSGD_eps1_z32_5.6"

# root_nodp = 'generated_mols/vae_nodp_z32_bs2048'
# root_eps_10 = 'generated_mols/VAE_eps1_z32_2048'
# root_eps_5 = 'generated_mols/vae_eps10_z32'
# root_eps_1 = "generated_mols/VAE_eps5_z32_2048"

# root_nodp = "generated_mols/gswgan_nodp_z32_pretrain_bs2048"
# root_eps_10 = 'generated_mols/gswgan_eps10_z32_pretrain_wgan'
# root_eps_5 = 'generated_mols/GSWGAN_eps_5_z32_pretrain_wgan_1.83'
# root_eps_1 = "generated_mols/GSWGAN_eps_1_z32_pretrain_wgan_7.07_bad"
#
# root_eps_1 = 'generated_mols/GSWGAN_eps_5_z32_pretrain_wgan_1.83'
# root_eps_5 = "generated_mols/GSWGAN_eps_1_z32_pretrain_wgan_7.07_bad"

# temp_root = 'generated_mols/gswgan_eps10_z32_pretrain_wgan'
# score_tmp = get_scores(load_smiles(temp_root))

score_nodp = get_scores(load_smiles(root_nodp))
scores_eps_1 = get_scores(load_smiles(root_eps_1))
scores_eps_5 = get_scores(load_smiles(root_eps_5))
scores_eps_10 = get_scores(load_smiles(root_eps_10))

# scores_gen = [score_molgan, score_graphvae, score_gan, score_vae, score_pate]
# plot_KDE_NoDP(score_orig, scores_gen)

scores_gen = [score_nodp, scores_eps_10, scores_eps_5, scores_eps_1]
plot_KDE_eps(score_orig, scores_gen, name='dp_ggan')
