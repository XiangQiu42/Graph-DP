#!/usr/bin/env/python
"""
Usage:
    evaluate.py --dataset zinc|qm9|cep

Options:
    -h --help                Show this screen.
    --dataset NAME           Dataset name: zinc, qm9
"""

import numpy as np
import argparse
import util_dir.graph_utils as g_utils


def get_config():
    parser = argparse.ArgumentParser()

    # Use either of these two datasets.
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--dataset', type=str, choices=['qm9', 'zinc'])
    config = parser.parse_args()

    config.dataset = 'qm9'
    config.qm9_dir = 'data/smiles/smiles_qm9.pkl'
    # config.qm9_dir = 'data/qm9_5k.smi'

    # generated root dir
    config.data_dir = "generated_mols/GSWGAN_eps_5_z32_pretrain_wgan_1.83"

    return config


if __name__ == '__main__':

    args = get_config()
    dataset_root = args.data_dir
    dataset = args.dataset
    root_qm9 = args.qm9_dir

    logpscorer, logp_score_per_molecule = g_utils.check_logp(dataset, dataset_root)
    qedscorer, qed_score_per_molecule = g_utils.check_qed(dataset, dataset_root)
    # novelty = g_utils.novelty_metric(dataset, dataset_root, root_qm9)
    # total, nonplanar = g_utils.check_planar(dataset_root)
    total, atom_counter, atom_per_molecule = g_utils.count_atoms(dataset, dataset_root)
    total, edge_type_counter, edge_type_per_molecule = g_utils.count_edge_type(dataset, dataset_root)
    total, shape_count, shape_count_per_molecule = g_utils.shape_count(dataset, dataset_root)
    total, tree_count = g_utils.check_cyclic(dataset, dataset_root)
    # sascorer, sa_score_per_molecule = g_utils.check_sascorer(dataset_root)
    total, validity = g_utils.check_validity(dataset, dataset_root)

    print("------------------------------------------")
    print("Metrics")
    print("------------------------------------------")
    print("total molecule")
    print(total)
    print("------------------------------------------")
    print("percentage of nonplanar:")
    # print(nonplanar / total)
    print("------------------------------------------")
    print("avg atom:")
    for atom_type, c in atom_counter.items():
        print(g_utils.dataset_info(dataset)['atom_types'][atom_type])
        print(c / total)
    print("standard deviation")
    print(np.std(atom_per_molecule, axis=0))
    print("------------------------------------------")
    print("avg edge_type:")
    for edge_type, c in edge_type_counter.items():
        print(edge_type + 1)
        print(c / total)
    print("standard deviation")
    print(np.std(edge_type_per_molecule, axis=0))
    print("------------------------------------------")
    print("avg shape:")
    for shape, c in zip(g_utils.geometry_numbers, shape_count):
        print(shape)
        print(c / total)
    print("standard deviation")
    print(np.std(shape_count_per_molecule, axis=0))
    print("------------------------------------------")
    print("percentage of tree:")
    print(tree_count / total)
    print("------------------------------------------")
    print("percentage of validity:")
    print(validity / total)
    print("------------------------------------------")
    print("avg sa_score:")
    # print(sascorer)
    print("standard deviation")
    # print(np.std(sa_score_per_molecule))
    print("------------------------------------------")
    print("avg logp_score:")
    print(logpscorer)
    print("standard deviation")
    print(np.std(logp_score_per_molecule))
    print("------------------------------------------")
    print("percentage of novelty:")
    # print(novelty)
    print("------------------------------------------")
    print("avg qed_score:")
    print(qedscorer)
    print("standard deviation")
    print(np.std(qed_score_per_molecule))
    print("------------------------------------------")
    print("uniqueness")
    print(g_utils.check_uniqueness(dataset, dataset_root))
    print("------------------------------------------")
    print("percentage of SSSR")
    print(g_utils.sssr_metric(dataset, dataset_root))
