# Changelog

## 1.4.0
- add ridge regression preprocessing [https://science.sciencemag.org/content/367/6480/eaay3224.abstract]((Park, 2020))
- require batches to feature a minimum of `neighbors_within_batch` cells each
- rework demo notebook, remove old preprint notebooks to [separate repository](https://github.com/Teichlab/bbknn_preprint)

## 1.3.12
- add packaging as a dependency in `setup.py`

## 1.3.11
- fix version parsing for newer versions of `anndata` and `packaging`

## 1.3.10
- store the computed graphs in `.obsp` if anndata is 0.7 or newer
- include additional parameter information in `.uns['neighbors']`
- return `.uns['neighbor']`-style parameter dictionary as third tuple element from `bbknn_pca_matrix`

## 1.3.9
- sklearn is actually scikit-learn for dependency purposes (D. DeTomaso)

## 1.3.8
- support umap-learn 0.4 connectivity output in a backward compatible manner (I. Driver)

## 1.3.7
- support for non-PCA dimensionality reductions on input (brought to attention by D. Cittaro)

## 1.3.6
- further updated scanpy logging compatibility tweak (F. Rost)

## 1.3.5
- update logging to match scanpy 1.4.4 standards

## 1.3.4
- rework `.obsm['X_pca']` check to be compatible with new structure (I. Virshup)

## 1.3.3
- add MANIFEST.in for conda purposes (J. Preußner)

## 1.3.2
- run trimming by default at 10 times the neighbour count for each cell
- remove scanpy dependency from `bbknn_pca_matrix`; rework setup dependencies as a result
- removal of `save_knn` option
- creation of diagnostic `extract_cell_connectivity` function for plotting connectivities

## 1.3.1
- metric sanity check typo fix

## 1.3.0
- optional faiss support
- default swap to annoy neighbours with angular as the metric, add annoy's `n_trees` parameter
- removal of distance scaling
- removal of `n_jobs` parameter - automatically parallelise cKDTree
- metric sanity checks

## 1.2.0
- `bbknn_pca_matrix` function
- code refactoring

## 1.1.0
- annoy support

## 1.0.0
- initial release