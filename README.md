# Batch balanced KNN

BBKNN is a fast and intuitive batch effect removal tool that can be directly used in the [scanpy](https://scanpy.readthedocs.io/en/latest/) workflow. It serves as an alternative to `scanpy.api.pp.neighbors()`, with both functions creating a neighbour graph for subsequent use in clustering, pseudotime and UMAP visualisation. The standard approach begins by identifying the k nearest neighbours for each individual cell across the entire data structure, with the candidates being subsequently transformed to exponentially related connectivities before serving as the basis for further analyses. If technical artifacts (be they because of differing data acquisition technologies, protocol alterations or even particularly severe operator effects) are present in the data, they will make it challenging to link corresponding cell types across different batches.

<p align="center"><img src="figures/batch1.png" alt="KNN" width="50%"></p>

As such, BBKNN actively combats this effect by splitting your data into batches and finding a smaller number of neighbours for each cell within each of the groups. This helps create connections between analogous cells in different batches without altering the counts or PCA space.

<p align="center"><img src="figures/batch2.png" alt="BBKNN" width="50%"></p>

## Installation

BBKNN depends on Cython, numpy, annoy and scanpy. The package is available on pip, and can be easily installed as follows:

	pip3 install bbknn

## Usage and Documentation

BBKNN has the option to immediately slot into the spot occupied by `scanpy.api.neighbors()` in the [Seurat-inspired scanpy workflow](https://nbviewer.jupyter.org/github/theislab/scanpy_usage/blob/master/170505_seurat/seurat.ipynb). It computes a batch aligned variant of the neighbourhood graph, with its uses within scanpy including clustering, diffusion map pseudotime inference and UMAP visualisation. The basic syntax to run BBKNN on scanpy's AnnData object (with PCA computed via `scanpy.api.tl.pca()`) is as follows:

	import bbknn
	bbknn.bbknn(adata)

You can provide which `adata.obs` column to use for batch discrimination via the `batch_key` parameter. This defaults to `'batch'`, which is created by scanpy when you merge multiple AnnData objects (e.g. if you were to import multiple samples separately and then concatenate them).

Alternately, you can just provide a PCA matrix with cells as rows and a matching vector of batch assignments for each of the cells and call BBKNN as follows (with `connectivities` being the primary graph output of interest):

	import bbknn
	distances, connectivities = bbknn.bbknn_pca_matrix(pca_matrix, batch_list)

If your dataset is large (e.g. 50,000-100,000 cells), consider setting `approx=True` in your BBKNN call to switch to the more computationally efficient approximate neighbour detection algorithm. An HTML render of the BBKNN function docstring, detailing all the parameters, can be accessed at [ReadTheDocs](https://bbknn.readthedocs.io/en/latest/).

## Example Notebooks

The repository also features Jupyter Notebooks capturing a range of biological and simulated examples of BBKNN use, along with comparisons to established batch correction methods. These analyses are explained in more detail in the [BBKNN preprint](https://www.biorxiv.org/content/early/2018/08/22/397042). All of the corresponding objects can be downloaded from [ftp://ngs.sanger.ac.uk/production/teichmann/BBKNN/](ftp://ngs.sanger.ac.uk/production/teichmann/BBKNN/)

- **[pancreas.ipynb](https://nbviewer.jupyter.org/github/Teichlab/bbknn/blob/master/examples/pancreas.ipynb) is the main demonstration, featuring in-depth annotation and a step by step description/comparison of BBKNN's available options.** [pancreas-2-mnnCorrect.ipynb](https://nbviewer.jupyter.org/github/Teichlab/bbknn/blob/master/examples/pancreas-2-mnnCorrect.ipynb) is a companion notebook that sees the same data processed with both the R original and third party Python reimplementation of mnnCorrect, while [pancreas-3-CCA.ipynb](https://nbviewer.jupyter.org/github/Teichlab/bbknn/blob/master/examples/pancreas-3-CCA.ipynb) processes the data with Seurat's MultiCCA and [pancreas-4-Scanorama.ipynb](https://nbviewer.jupyter.org/github/Teichlab/bbknn/blob/master/examples/pancreas-4-Scanorama.ipynb) does the same with Scanorama.
- [pbmc.ipynb](https://nbviewer.jupyter.org/github/Teichlab/bbknn/blob/master/examples/pbmc.ipynb) and [mouse.ipynb](https://nbviewer.jupyter.org/github/Teichlab/bbknn/blob/master/examples/mouse.ipynb) capture the core of the 10X protocol variant PBMC merging and integrative analysis of murine cell atlases respectively. They are annotated in less depth than the pancreas notebooks.
- [simulation.ipynb](https://nbviewer.jupyter.org/github/Teichlab/bbknn/blob/master/examples/simulation.ipynb) applies BBKNN to simulated data with a known ground truth, and demonstrates the utility of graph trimming by introducing an unrelated cell population. This simulated data is then used to benchmark BBKNN against mnnCorrect, CCA and Scanorama in [benchmark.ipynb](https://nbviewer.jupyter.org/github/Teichlab/bbknn/blob/master/examples/benchmark.ipynb), and then perform a similar test on larger datasets for BBKNN and Scanorama only in [benchmark2.ipynb](https://nbviewer.jupyter.org/github/Teichlab/bbknn/blob/master/examples/benchmark2.ipynb).

## Murine Atlas Integration Exploration

The murine objects, created during an integrative analysis detailed in [the preprint](https://www.biorxiv.org/content/early/2018/08/22/397042), can be downloaded from [ftp://ngs.sanger.ac.uk/production/teichmann/BBKNN/MouseAtlas.zip](ftp://ngs.sanger.ac.uk/production/teichmann/BBKNN/MouseAtlas.zip) and easily explored. A dedicated exploration notebook with examples and explanations is provided at [mouse-exploratory-visualisation.ipynb](https://nbviewer.jupyter.org/github/Teichlab/bbknn/blob/master/examples/mouse-exploratory-visualisation.ipynb). This includes the extraction of modules of correlated transcription factors and an interactive visualisation where hovering reveals the gene name.