# Graph generative model with Graph level DP

This repository contains the implementation of our attention for 
ensuring a Graph Generative Model with Graph Differential Privacy(Graph DP).

## Data Download and Preprocessing
Just simply run a bash script. You can find such a bash script in the data directory. If you see warnings or even errors, please just don't worry.

If you wish to use the QM9 dataset, you can skip the data downloading and directly proceed to data preprocessing.
```bash
bash data/download_dataset.sh
```
Then run the python script below.
```bash
python data/sparse_molecular_dataset.py
```

## DP-GGAN
Simply run the following command to train.
```bash
python main_gan.py
```

## DP-GVAE
Simply run the following command to train.
```bash
python main_vae.py
```

## Graph-PATE
Simply run the following command to train.
```bash
python main_pate.py
```

## Acknowledgements

Our implementation uses the source code from the following repositories:

* [Opacus](https://github.com/pytorch/opacus)

* [MolGAN (Tensorflow)](https://github.com/nicola-decao/MolGAN)

* [PyTorch Implementation of MolGAN (Pytorch)](https://github.com/ZhenyueQin/Implementation-MolGAN-PyTorch)

* [G-PATE (Tensorflow)](https://github.com/AI-secure/G-PATE)
