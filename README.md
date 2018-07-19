# Batch balanced KNN

BBKNN is a fast and intuitive batch effect removal tool for direct use in the [scanpy](https://scanpy.readthedocs.io/en/latest/) workflow. It serves as an alternative to `scanpy.api.pp.neighbors()`, with both functions creating a neighbour graph for subsequent use in clustering, pseudotime and UMAP visualisation. The standard approach begins by identifying the k nearest neighbours for each individual cell across the entire data structure, with the candidates being subsequently transformed to exponentially related connectivities before serving as the basis for further analyses. If technical artifacts (be they because of differing data acquisition technologies, protocol alterations or even particularly severe operator effects) are present in the data, they will make it challenging to link corresponding cell types across different batches.

<p align="center"><img src="figures/batch1.png" alt="KNN" width="50%"></p>

As such, BBKNN actively combats this effect by splitting your data into batches and finding a smaller number of neighbours for each cell within each of the groups. This helps create connections between analogous cells in different batches without altering the counts or PCA space.

<p align="center"><img src="figures/batch2.png" alt="BBKNN" width="50%"></p>

## Installation

BBKNN is available on pip, and can be easily installed as follows:

	pip3 install bbknn

The package depends on Cython, numpy and scanpy.