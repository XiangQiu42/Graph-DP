import os
from rdkit import Chem
from rdkit.Chem import Draw

csv_path = "data/qm9_5k.smi"

data = []
with open(csv_path, "r") as f:
    for line in f.readlines()[1:]:
        data.append(line)

# Let's look at a molecule of the dataset

for i in range(22, 44):
    smiles = data[i]
    print(f"{i}: SMILES:", smiles)
    molecule = Chem.MolFromSmiles(smiles)
    print("Num heavy atoms:", molecule.GetNumHeavyAtoms())
    path = os.path.join('generated_smiles', f'{i}.png')
    Draw.MolToFile(molecule, path)