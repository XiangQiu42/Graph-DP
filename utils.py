import rdkit
from sklearn.metrics import classification_report as sk_classification_report
from sklearn.metrics import confusion_matrix
from pysmiles import read_smiles

import pickle
import gzip
from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import Crippen
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

import math
from collections import defaultdict, deque
import numpy as np

from util_dir.utils_io import random_string

NP_model = pickle.load(gzip.open('data/NP_score.pkl.gz'))
SA_model = {i[j]: float(i[0]) for i in pickle.load(gzip.open('data/SA_score.pkl.gz')) for j in range(1, len(i))}


class MolecularMetrics(object):

    @staticmethod
    def _avoid_sanitization_error(op):
        try:
            return op()
        except ValueError:
            return None

    @staticmethod
    def remap(x, x_min, x_max):
        return (x - x_min) / (x_max - x_min)

    @staticmethod
    def valid_lambda(x):
        return x is not None and Chem.MolToSmiles(x) != ''

    @staticmethod
    def valid_lambda_special(x):
        s = Chem.MolToSmiles(x) if x is not None else ''
        return x is not None and '*' not in s and '.' not in s and s != ''

    @staticmethod
    def valid_scores(mols):
        return np.array(list(map(MolecularMetrics.valid_lambda_special, mols)), dtype=np.float32)

    @staticmethod
    def valid_filter(mols):
        return list(filter(MolecularMetrics.valid_lambda, mols))

    @staticmethod
    def valid_total_score(mols):
        return np.array(list(map(MolecularMetrics.valid_lambda, mols)), dtype=np.float32).mean()

    @staticmethod
    def novel_scores(mols, data):
        return np.array(
            list(map(lambda x: MolecularMetrics.valid_lambda(x) and Chem.MolToSmiles(x) not in data.smiles, mols)))

    @staticmethod
    def novel_filter(mols, data):
        return list(filter(lambda x: MolecularMetrics.valid_lambda(x) and Chem.MolToSmiles(x) not in data.smiles, mols))

    @staticmethod
    def novel_total_score(mols, data):
        return MolecularMetrics.novel_scores(MolecularMetrics.valid_filter(mols), data).mean()

    @staticmethod
    def unique_scores(mols):
        smiles = list(map(lambda x: Chem.MolToSmiles(x) if MolecularMetrics.valid_lambda(x) else '', mols))
        return np.clip(
            0.75 + np.array(list(map(lambda x: 1 / smiles.count(x) if x != '' else 0, smiles)), dtype=np.float32), 0, 1)

    @staticmethod
    def unique_total_score(mols):
        v = MolecularMetrics.valid_filter(mols)
        s = set(map(lambda x: Chem.MolToSmiles(x), v))
        return 0 if len(v) == 0 else len(s) / len(v)

    # @staticmethod
    # def novel_and_unique_total_score(mols, data):
    #     return ((MolecularMetrics.unique_scores(mols) == 1).astype(float) * MolecularMetrics.novel_scores(mols,
    #                                                                                                       data)).sum()
    #
    # @staticmethod
    # def reconstruction_scores(data, model, session, sample=False):
    #
    #     m0, _, _, a, x, _, f, _, _ = data.next_validation_batch()
    #     feed_dict = {model.edges_labels: a, model.nodes_labels: x, model.node_features: f, model.training: False}
    #
    #     try:
    #         feed_dict.update({model.variational: False})
    #     except AttributeError:
    #         pass
    #
    #     n, e = session.run([model.nodes_gumbel_argmax, model.edges_gumbel_argmax] if sample else [
    #         model.nodes_argmax, model.edges_argmax], feed_dict=feed_dict)
    #
    #     n, e = np.argmax(n, axis=-1), np.argmax(e, axis=-1)
    #
    #     m1 = [data.matrices2mol(n_, e_, strict=True) for n_, e_ in zip(n, e)]
    #
    #     return np.mean([float(Chem.MolToSmiles(m0_) == Chem.MolToSmiles(m1_)) if m1_ is not None else 0
    #             for m0_, m1_ in zip(m0, m1)])

    @staticmethod
    def natural_product_scores(mols, norm=False):

        # calculating the score
        scores = [sum(NP_model.get(bit, 0)
                      for bit in Chem.rdMolDescriptors.GetMorganFingerprint(mol,
                                                                            2).GetNonzeroElements()) / float(
            mol.GetNumAtoms()) if mol is not None else None
                  for mol in mols]

        # preventing score explosion for exotic molecules
        scores = list(map(lambda score: score if score is None else (
            4 + math.log10(score - 4 + 1) if score > 4 else (
                -4 - math.log10(-4 - score + 1) if score < -4 else score)), scores))

        scores = np.array(list(map(lambda x: -4 if x is None else x, scores)))
        scores = np.clip(MolecularMetrics.remap(scores, -3, 1), 0.0, 1.0) if norm else scores

        return scores

    @staticmethod
    def quantitative_estimation_druglikeness_scores(mols, norm=False):
        return np.array(list(map(lambda x: 0 if x is None else x, [
            MolecularMetrics._avoid_sanitization_error(lambda: QED.qed(mol)) if mol is not None else None for mol in
            mols])))

    @staticmethod
    def water_octanol_partition_coefficient_scores(mols, norm=False):
        scores = [MolecularMetrics._avoid_sanitization_error(lambda: Crippen.MolLogP(mol)) if mol is not None else None
                  for mol in mols]
        scores = np.array(list(map(lambda x: -3 if x is None else x, scores)))
        scores = np.clip(MolecularMetrics.remap(scores, -2.12178879609, 6.0429063424), 0.0, 1.0) if norm else scores

        return scores

    @staticmethod
    def _compute_SAS(mol):
        fp = Chem.rdMolDescriptors.GetMorganFingerprint(mol, 2)
        fps = fp.GetNonzeroElements()
        score1 = 0.
        nf = 0
        # for bitId, v in fps.items():
        for bitId, v in fps.items():
            nf += v
            sfp = bitId
            score1 += SA_model.get(sfp, -4) * v
        score1 /= nf

        # features score
        nAtoms = mol.GetNumAtoms()
        nChiralCenters = len(Chem.FindMolChiralCenters(
            mol, includeUnassigned=True))
        ri = mol.GetRingInfo()
        nSpiro = Chem.rdMolDescriptors.CalcNumSpiroAtoms(mol)
        nBridgeheads = Chem.rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        nMacrocycles = 0
        for x in ri.AtomRings():
            if len(x) > 8:
                nMacrocycles += 1

        sizePenalty = nAtoms ** 1.005 - nAtoms
        stereoPenalty = math.log10(nChiralCenters + 1)
        spiroPenalty = math.log10(nSpiro + 1)
        bridgePenalty = math.log10(nBridgeheads + 1)
        macrocyclePenalty = 0.

        # ---------------------------------------
        # This differs from the paper, which defines:
        #  macrocyclePenalty = math.log10(nMacrocycles+1)
        # This form generates better results when 2 or more macrocycles are present
        if nMacrocycles > 0:
            macrocyclePenalty = math.log10(2)

        score2 = 0. - sizePenalty - stereoPenalty - \
                 spiroPenalty - bridgePenalty - macrocyclePenalty

        # correction for the fingerprint density
        # not in the original publication, added in version 1.1
        # to make highly symmetrical molecules easier to synthetise
        score3 = 0.
        if nAtoms > len(fps):
            score3 = math.log(float(nAtoms) / len(fps)) * .5

        sascore = score1 + score2 + score3

        # need to transform "raw" value into scale between 1 and 10
        min = -4.0
        max = 2.5
        sascore = 11. - (sascore - min + 1) / (max - min) * 9.
        # smooth the 10-end
        if sascore > 8.:
            sascore = 8. + math.log(sascore + 1. - 9.)
        if sascore > 10.:
            sascore = 10.0
        elif sascore < 1.:
            sascore = 1.0

        return sascore

    @staticmethod
    def synthetic_accessibility_score_scores(mols, norm=False):
        scores = [MolecularMetrics._compute_SAS(mol) if mol is not None else None for mol in mols]
        scores = np.array(list(map(lambda x: 10 if x is None else x, scores)))
        scores = np.clip(MolecularMetrics.remap(scores, 5, 1.5), 0.0, 1.0) if norm else scores

        return scores

    @staticmethod
    def diversity_scores(mols, data):
        rand_mols = np.random.choice(data.data, 100)
        fps = [Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 4, nBits=2048) for mol in rand_mols]

        scores = np.array(
            list(map(lambda x: MolecularMetrics.__compute_diversity(x, fps) if x is not None else 0, mols)))
        scores = np.clip(MolecularMetrics.remap(scores, 0.9, 0.945), 0.0, 1.0)

        return scores

    @staticmethod
    def __compute_diversity(mol, fps):
        ref_fps = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 4, nBits=2048)
        dist = DataStructs.BulkTanimotoSimilarity(ref_fps, fps, returnDistance=True)
        score = np.mean(dist)
        return score

    @staticmethod
    def drugcandidate_scores(mols, data):

        scores = (MolecularMetrics.constant_bump(
            MolecularMetrics.water_octanol_partition_coefficient_scores(mols, norm=True), 0.210,
            0.945) + MolecularMetrics.synthetic_accessibility_score_scores(mols,
                                                                           norm=True) + MolecularMetrics.novel_scores(
            mols, data) + (1 - MolecularMetrics.novel_scores(mols, data)) * 0.3) / 4

        return scores

    @staticmethod
    def constant_bump(x, x_low, x_high, decay=0.025):
        return np.select(condlist=[x <= x_low, x >= x_high],
                         choicelist=[np.exp(- (x - x_low) ** 2 / decay),
                                     np.exp(- (x - x_high) ** 2 / decay)],
                         default=np.ones_like(x))


def mols2grid_image(mols, molsPerRow):
    mols = [e if e is not None else Chem.RWMol() for e in mols]

    for mol in mols:
        AllChem.Compute2DCoords(mol)

    return Draw.MolsToGridImage(mols, molsPerRow=molsPerRow, subImgSize=(150, 150))


def classification_report(data, model, session, sample=False):
    _, _, _, a, x, _, f, _, _ = data.next_validation_batch()

    n, e = session.run([model.nodes_gumbel_argmax, model.edges_gumbel_argmax] if sample else [
        model.nodes_argmax, model.edges_argmax], feed_dict={model.edges_labels: a, model.nodes_labels: x,
                                                            model.node_features: f, model.training: False,
                                                            model.variational: False})
    n, e = np.argmax(n, axis=-1), np.argmax(e, axis=-1)

    y_true = e.flatten()
    y_pred = a.flatten()
    target_names = [str(Chem.rdchem.BondType.values[int(e)]) for e in data.bond_decoder_m.values()]

    print('######## Classification Report ########\n')
    print(sk_classification_report(y_true, y_pred, labels=list(range(len(target_names))),
                                   target_names=target_names))

    print('######## Confusion Matrix ########\n')
    print(confusion_matrix(y_true, y_pred, labels=list(range(len(target_names)))))

    y_true = n.flatten()
    y_pred = x.flatten()
    target_names = [Chem.Atom(e).GetSymbol() for e in data.atom_decoder_m.values()]

    print('######## Classification Report ########\n')
    print(sk_classification_report(y_true, y_pred, labels=list(range(len(target_names))),
                                   target_names=target_names))

    print('\n######## Confusion Matrix ########\n')
    print(confusion_matrix(y_true, y_pred, labels=list(range(len(target_names)))))


def reconstructions(data, model, session, batch_dim=10, sample=False):
    m0, _, _, a, x, _, f, _, _ = data.next_train_batch(batch_dim)

    n, e = session.run([model.nodes_gumbel_argmax, model.edges_gumbel_argmax] if sample else [
        model.nodes_argmax, model.edges_argmax], feed_dict={model.edges_labels: a, model.nodes_labels: x,
                                                            model.node_features: f, model.training: False,
                                                            model.variational: False})
    n, e = np.argmax(n, axis=-1), np.argmax(e, axis=-1)

    m1 = np.array([e if e is not None else Chem.RWMol() for e in [data.matrices2mol(n_, e_, strict=True)
                                                                  for n_, e_ in zip(n, e)]])

    mols = np.vstack((m0, m1)).T.flatten()

    return mols


def samples(data, model, session, embeddings, sample=False):
    n, e = session.run([model.nodes_gumbel_argmax, model.edges_gumbel_argmax] if sample else [
        model.nodes_argmax, model.edges_argmax], feed_dict={
        model.embeddings: embeddings, model.training: False})
    n, e = np.argmax(n, axis=-1), np.argmax(e, axis=-1)

    mols = [data.matrices2mol(n_, e_, strict=True) for n_, e_ in zip(n, e)]

    return mols


def all_scores(mols, data, norm=False, reconstruction=False):
    m0 = {k: list(filter(lambda e: e is not None, v)) for k, v in {
        'NP': MolecularMetrics.natural_product_scores(mols, norm=norm),
        'QED': MolecularMetrics.quantitative_estimation_druglikeness_scores(mols),
        'Solute': MolecularMetrics.water_octanol_partition_coefficient_scores(mols, norm=norm),
        'SA': MolecularMetrics.synthetic_accessibility_score_scores(mols, norm=norm),
        'diverse': MolecularMetrics.diversity_scores(mols, data),
        'drugcand': MolecularMetrics.drugcandidate_scores(mols, data)}.items()}

    m1 = {'valid': MolecularMetrics.valid_total_score(mols) * 100,
          'unique': MolecularMetrics.unique_total_score(mols) * 100,
          'novel': MolecularMetrics.novel_total_score(mols, data) * 100}

    return m0, m1


def save_mol_img(mols, f_name='tmp.png', is_test=False):
    orig_f_name = f_name
    for a_mol in mols:
        try:
            if Chem.MolToSmiles(a_mol) is not None:
                # print('Generating molecule')

                if is_test:
                    f_name = orig_f_name
                    f_split = f_name.split('.')
                    f_split[-1] = random_string() + '.' + f_split[-1]
                    f_name = ''.join(f_split)

                rdkit.Chem.Draw.MolToFile(a_mol, f_name)
                a_smi = Chem.MolToSmiles(a_mol)
                mol_graph = read_smiles(a_smi)

                break

                # if not is_test:
                #     break
        except:
            continue


def visualize_mol(path, new_mol):
    AllChem.Compute2DCoords(new_mol)
    print(f'Visualized generated model on {path}')
    Draw.MolToFile(new_mol, path)


def dump(file_name, content):
    print(f'We dump {len(content)} graphs in {file_name}')
    with open(file_name, 'wb') as out_file:
        pickle.dump(content, out_file, pickle.HIGHEST_PROTOCOL)


def get_device_id(id, num_discriminators, num_gpus):
    partitions = np.linspace(0, 1, num_gpus, endpoint=False)[1:]
    device_id = 0
    for p in partitions:
        if id <= num_discriminators * p:
            break
        device_id += 1
    return device_id


def exp_mov_avg(Gs, G, alpha=0.999, global_step=999):
    """
    Exponential moving average of generator parameters
    :param Gs:
    :param G:
    :param alpha:
    :param global_step:
    :return:
    """
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(Gs.parameters(), G.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


def discriminators_EMA(odd_d_acc, d_acc, alpha=0.99, global_step=999):
    """
    This mehtod is motivated by paper https://openreview.net/forum?id=QEmn_Hvh7j8
    """
    if odd_d_acc == 0:
        return d_acc

    alpha = min(alpha, 1 - 1 / (global_step + 1))

    new_average = (1.0 - alpha) * d_acc + alpha * odd_d_acc
    return new_average


def get_accuracy(label, y_prob):
    assert y_prob.ndim == 1
    y_prob = y_prob > 0.5
    return (y_prob == label).sum().item() / y_prob.size(0)


# #  some help functions
# from rdkit.Chem import rdmolops
#
# geometry_numbers = [3, 4, 5, 6]  # triangle, square, pentagen, hexagon
#
# # bond mapping
# bond_dict = {'SINGLE': 0, 'DOUBLE': 1, 'TRIPLE': 2, "AROMATIC": 3}
# number_to_bond = {0: Chem.rdchem.BondType.SINGLE, 1: Chem.rdchem.BondType.DOUBLE,
#                   2: Chem.rdchem.BondType.TRIPLE, 3: Chem.rdchem.BondType.AROMATIC}
#
#
# def dataset_info(dataset):  # qm9, zinc, cep
#     if dataset == 'qm9':
#         return {'atom_types': ["H", "C", "N", "O", "F"],
#                 'maximum_valence': {0: 1, 1: 4, 2: 3, 3: 2, 4: 1},
#                 'number_to_atom': {0: "H", 1: "C", 2: "N", 3: "O", 4: "F"},
#                 'bucket_sizes': np.array(list(range(4, 28, 2)) + [29])
#                 }
#     elif dataset == 'zinc':
#         return {'atom_types': ['Br1(0)', 'C4(0)', 'Cl1(0)', 'F1(0)', 'H1(0)', 'I1(0)',
#                                'N2(-1)', 'N3(0)', 'N4(1)', 'O1(-1)', 'O2(0)', 'S2(0)', 'S4(0)', 'S6(0)'],
#                 'maximum_valence': {0: 1, 1: 4, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 3, 8: 4, 9: 1, 10: 2, 11: 2, 12: 4,
#                                     13: 6, 14: 3},
#                 'number_to_atom': {0: 'Br', 1: 'C', 2: 'Cl', 3: 'F', 4: 'H', 5: 'I', 6: 'N', 7: 'N', 8: 'N', 9: 'O',
#                                    10: 'O', 11: 'S', 12: 'S', 13: 'S'},
#                 'bucket_sizes': np.array(
#                     [28, 31, 33, 35, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 53, 55, 58, 84])
#                 }
#
#     elif dataset == "cep":
#         return {'atom_types': ["C", "S", "N", "O", "Se", "Si"],
#                 'maximum_valence': {0: 4, 1: 2, 2: 3, 3: 2, 4: 2, 5: 4},
#                 'number_to_atom': {0: "C", 1: "S", 2: "N", 3: "O", 4: "Se", 5: "Si"},
#                 'bucket_sizes': np.array([25, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 43, 46])
#                 }
#     else:
#         print("the datasets in use are qm9|zinc|cep")
#         exit(1)
#
#
# def need_kekulize(mol):
#     for bond in mol.GetBonds():
#         if bond_dict[str(bond.GetBondType())] >= 3:
#             return True
#     return False
#
#
# def onehot(idx, len):
#     z = [0 for _ in range(len)]
#     z[idx] = 1
#     return z
#
#
# def mol_to_graph(mol, dataset_name):
#     if mol is None:
#         return None, None
#
#     # ?????????????????????????????????????????
#     # Kekulize it
#     if need_kekulize(mol):
#         rdmolops.Kekulize(mol)
#         if mol is None:
#             return None, None
#
#     # remove stereo information, such as inward and outward edges
#     Chem.RemoveStereochemistry(mol)
#
#     edges = []
#     nodes = []
#     for bond in mol.GetBonds():
#         edges.append((bond.GetBeginAtomIdx(), bond_dict[str(bond.GetBondType())], bond.GetEndAtomIdx()))
#         assert bond_dict[str(bond.GetBondType())] != 3
#     for atom in mol.GetAtoms():
#         if dataset_name == 'qm9' or dataset_name == "cep":
#             nodes.append(onehot(dataset_info(dataset_name)['atom_types'].index(atom.GetSymbol()),
#                                 len(dataset_info(dataset_name)['atom_types'])))
#         elif dataset_name == 'zinc':  # transform using "<atom_symbol><valence>(<charge>)"  notation
#             symbol = atom.GetSymbol()
#             valence = atom.GetTotalValence()
#             charge = atom.GetFormalCharge()
#             atom_str = "%s%i(%i)" % (symbol, valence, charge)
#
#             if atom_str not in dataset_info(dataset_name)['atom_types']:
#                 print('unrecognized atom type %s' % atom_str)
#                 return [], []
#
#             nodes.append(
#                 onehot(dataset_info(dataset_name)['atom_types'].index(atom_str),
#                        len(dataset_info(dataset_name)['atom_types'])))
#
#     return nodes, edges
#
#
# def count_edge_type(mols, dataset_name):
#     # if generated:
#     #     filename = dataset_root
#     # else:
#     #     filename = 'all_smiles_%s.pkl' % dataset
#     # with open(filename, 'rb') as f:
#     #     all_smiles = set(pickle.load(f))
#     cnt = 0
#     counter = defaultdict(int)
#     edge_type_per_molecule = []
#     for mol in mols:
#         # nodes, edges = to_graph(smiles, dataset)
#
#         try:
#             nodes, edges = mol_to_graph(mol, dataset_name)
#         except:
#             continue
#
#         if (not nodes) or not edges:
#             continue
#
#         cnt += 1
#
#         edge_type_this_molecule = [0] * len(bond_dict)
#         for edge in edges:
#             edge_type = edge[1]
#             edge_type_this_molecule[edge_type] += 1
#             counter[edge_type] += 1
#         edge_type_per_molecule.append(edge_type_this_molecule)
#     total_sum = 0
#     return cnt, counter, edge_type_per_molecule
#
#
# def count_atoms(mols, dataset_name):
#     # with open(dataset_root, 'rb') as f:
#     #     all_smiles = set(pickle.load(f))
#     cnt = 0
#     counter = defaultdict(int)
#     atom_count_per_molecule = []  # record the counts for each molecule
#     for mol in mols:
#         try:
#             nodes, edges = mol_to_graph(mol, dataset_name)
#         except:
#             continue
#
#         if (not nodes) or not edges:
#             continue
#
#         cnt += 1
#
#         atom_count_this_molecule = [0] * len(dataset_info(dataset_name)['atom_types'])
#         for node in nodes:
#             atom_type = np.argmax(node)
#             atom_count_this_molecule[atom_type] += 1
#             counter[atom_type] += 1
#         atom_count_per_molecule.append(atom_count_this_molecule)
#     total_sum = 0
#
#     return cnt, counter, atom_count_per_molecule
#
#
# def count_shape(mols, dataset_name, remove_print=False, all_smiles=None):
#     cnt = 0
#     geometry_counts = [0] * len(geometry_numbers)
#     geometry_counts_per_molecule = []  # record the geometry counts for each molecule
#     for mol in mols:
#         # nodes, edges = to_graph(smiles, dataset)
#
#         try:
#             nodes, edges = mol_to_graph(mol, dataset_name)
#         except:
#             continue
#
#         if (not nodes) or not edges:
#             continue
#
#         cnt += 1
#
#         if len(edges) <= 0:
#             continue
#         # new_mol = Chem.MolFromSmiles(smiles)
#
#         ssr = Chem.GetSymmSSSR(mol)
#         counts_for_molecule = [0] * len(geometry_numbers)
#         for idx in range(len(ssr)):
#             ring_len = len(list(ssr[idx]))
#             if ring_len in geometry_numbers:
#                 geometry_counts[geometry_numbers.index(ring_len)] += 1
#                 counts_for_molecule[geometry_numbers.index(ring_len)] += 1
#         geometry_counts_per_molecule.append(counts_for_molecule)
#
#     return cnt, geometry_counts, geometry_counts_per_molecule
