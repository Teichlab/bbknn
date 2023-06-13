"""Batch balanced KNN"""
__version__ = "1.5.1"

import pandas as pd
import numpy as np
import scipy
import sys
from sklearn.linear_model import Ridge
try:
	from scanpy import logging as logg
except ImportError:
	pass

from . import matrix

def bbknn(adata, batch_key='batch', use_rep='X_pca', copy=False, **kwargs):
	'''
	Batch balanced KNN, altering the KNN procedure to identify each cell's top neighbours in
	each batch separately instead of the entire cell pool with no accounting for batch. 
	The nearest neighbours for each batch are then merged to create a final list of 
	neighbours for the cell.
	Aligns batches in a quick and lightweight manner.
	For use in the scanpy workflow as an alternative to ``scanpy.pp.neighbors()``.

	Input
	-----
	adata : ``AnnData``
		Needs your dimensionality reduction of choice computed and stored in ``.obsm``.
	batch_key : ``str``, optional (default: "batch")
		``adata.obs`` column name discriminating between your batches.
	neighbors_within_batch : ``int``, optional (default: 3)
		How many top neighbours to report for each batch; total number of neighbours in 
		the initial k-nearest-neighbours computation will be this number times the number 
		of batches. This then serves as the basis for the construction of a symmetrical 
		matrix of connectivities.
	use_rep : ``str``, optional (default: "X_pca")
		The dimensionality reduction in ``.obsm`` to use for neighbour detection. Defaults to PCA.
	n_pcs : ``int``, optional (default: 50)
		How many dimensions (in case of PCA, principal components) to use in the analysis.
	trim : ``int`` or ``None``, optional (default: ``None``)
		Trim the neighbours of each cell to these many top connectivities. May help with
		population independence and improve the tidiness of clustering. The lower the value the
		more independent the individual populations, at the cost of more conserved batch effect.
		If ``None``, sets the parameter value automatically to 10 times ``neighbors_within_batch`` 
		times the number of batches. Set to 0 to skip.
	approx : ``bool``, optional (default: ``True``)
		If ``True``, use approximate neighbour finding - annoy or pyNNDescent. This results 
		in a quicker run time for large datasets while also potentially increasing the degree of 
		batch correction.
	use_annoy : ``bool``, optional (default: ``True``)
		Only used when ``approx=True``. If ``True``, will use annoy for neighbour finding. If
		``False``, will use pyNNDescent instead.
	annoy_n_trees : ``int``, optional (default: 10)
		Only used with annoy neighbour identification. The number of trees to construct in the 
		annoy forest. More trees give higher precision when querying, at the cost of increased 
		run time and resource intensity.
	pynndescent_n_neighbors : ``int``, optional (default: 30)
		Only used with pyNNDescent neighbour identification. The number of neighbours to include
		in the approximate neighbour graph. More neighbours give higher precision when querying, 
		at the cost of increased run time and resource intensity.
	pynndescent_random_state : ``int``, optional (default: 0)
		Only used with pyNNDescent neighbour identification. The RNG seed to use when creating 
		the graph.
	use_faiss : ``bool``, optional (default: ``True``)
		If ``approx=False`` and the metric is "euclidean", use the faiss package to compute
		nearest neighbours if installed. This improves performance at a minor cost to numerical
		precision as faiss operates on float32.
	metric : ``str`` or ``sklearn.neighbors.DistanceMetric`` or ``types.FunctionType``, optional (default: "euclidean")
		What distance metric to use. The options depend on the choice of neighbour algorithm.
		
		"euclidean", the default, is always available.
		
		Annoy supports "angular", "manhattan" and "hamming".
		
		PyNNDescent supports metrics listed in ``pynndescent.distances.named_distances``
		and custom functions, including compiled Numba code.
		
		>>> pynndescent.distances.named_distances.keys()
		dict_keys(['euclidean', 'l2', 'sqeuclidean', 'manhattan', 'taxicab', 'l1', 'chebyshev', 'linfinity', 
		'linfty', 'linf', 'minkowski', 'seuclidean', 'standardised_euclidean', 'wminkowski', 'weighted_minkowski', 
		'mahalanobis', 'canberra', 'cosine', 'dot', 'correlation', 'hellinger', 'haversine', 'braycurtis', 'spearmanr', 
		'kantorovich', 'wasserstein', 'tsss', 'true_angular', 'hamming', 'jaccard', 'dice', 'matching', 'kulsinski', 
		'rogerstanimoto', 'russellrao', 'sokalsneath', 'sokalmichener', 'yule'])
		
		KDTree supports members of the ``sklearn.neighbors.KDTree.valid_metrics`` list, or parameterised
		``sklearn.neighbors.DistanceMetric`` `objects
		<https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html>`_:

		>>> sklearn.neighbors.KDTree.valid_metrics
		['p', 'chebyshev', 'cityblock', 'minkowski', 'infinity', 'l2', 'euclidean', 'manhattan', 'l1']
	set_op_mix_ratio : ``float``, optional (default: 1)
		UMAP connectivity computation parameter, float between 0 and 1, controlling the
		blend between a connectivity matrix formed exclusively from mutual nearest neighbour
		pairs (0) and a union of all observed neighbour relationships with the mutual pairs
		emphasised (1)
	local_connectivity : ``int``, optional (default: 1)
		UMAP connectivity computation parameter, how many nearest neighbors of each cell
		are assumed to be fully connected (and given a connectivity value of 1)
	copy : ``bool``, optional (default: ``False``)
		If ``True``, return a copy instead of writing to the supplied adata.
	'''
	start = logg.info('computing batch balanced neighbors')
	adata = adata.copy() if copy else adata
	#basic sanity checks to begin
	#is our batch key actually present in the object?
	if batch_key not in adata.obs:
		raise ValueError("Batch key '"+batch_key+"' not present in `adata.obs`.")
	#do we have a computed PCA?
	if use_rep not in adata.obsm.keys():
		raise ValueError("Did not find "+use_rep+" in `.obsm.keys()`. You need to compute it first.")
	#prepare bbknn.matrix.bbknn input
	pca = adata.obsm[use_rep]
	batch_list = adata.obs[batch_key].values
	#call BBKNN proper, telling it to use scanpy logging for its internal things
	bbknn_out = matrix.bbknn(pca=pca, batch_list=batch_list, scanpy_logging=True, **kwargs)
	#store the parameters in .uns['neighbors']['params'], add use_rep and batch_key
	adata.uns['neighbors'] = {}
	adata.uns['neighbors']['params'] = bbknn_out[2]
	adata.uns['neighbors']['params']['use_rep'] = use_rep
	adata.uns['neighbors']['params']['bbknn']['batch_key'] = batch_key
	#store the graphs in an anndata 0.7.0+ compliant manner
	adata.obsp['distances'] = bbknn_out[0]
	adata.obsp['connectivities'] = bbknn_out[1]
	adata.uns['neighbors']['distances_key'] = 'distances'
	adata.uns['neighbors']['connectivities_key'] = 'connectivities'
	logg.info('	finished', time=start,
		deep=('added to `.uns[\'neighbors\']`\n'
		'	`.obsp[\'distances\']`, distances for each pair of neighbors\n'
		'	`.obsp[\'connectivities\']`, weighted adjacency matrix'))
	return adata if copy else None

def ridge_regression(adata, batch_key, confounder_key=[], chunksize=1e8, copy=False, **kwargs):
	'''
	Perform ridge regression on scaled expression data, accepting both technical and 
	biological categorical variables. The effect of the technical variables is removed 
	while the effect of the biological variables is retained. This is a preprocessing 
	step that can aid BBKNN integration `(Park, 2020) <https://science.sciencemag.org/content/367/6480/eaay3224.abstract>`_.
	
	Alters the object's ``.X`` to be the regression residuals, and creates ``.layers['X_explained']`` 
	with the expression explained by the technical effect.
	
	Input
	-----
	adata : ``AnnData``
		Needs scaled data in ``.X``.
	batch_key : ``list``
		A list of categorical ``.obs`` columns to regress out as technical effects.
	confounder_key : ``list``, optional (default: ``[]``)
		A list of categorical ``.obs`` columns to retain as biological effects.
	chunksize : ``int``, optional (default: 1e8)
		How many elements of the expression matrix to process at a time. Potentially useful 
		to manage memory use for larger datasets.
	copy : ``bool``, optional (default: ``False``)
		If ``True``, return a copy instead of writing to the supplied adata.
	kwargs
		Any arguments to pass to `Ridge <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html>`_.
	'''
	start = logg.info('computing ridge regression')
	adata = adata.copy() if copy else adata
	#just in case the arguments are not provided as lists, convert them to such
	#as they need to be lists for downstream application
	if not isinstance(batch_key, list):
		batch_key = [batch_key]
	if not isinstance(confounder_key, list):
		confounder_key = [confounder_key]
	
	#construct a helper representation of the batch and biological variables
	#as a data frame with one row per cell, with columns specifying the various batch/biological categories
	#with values of 1 where the cell is of the category and 0 otherwise (dummy)
	#and subsequently identify which of the data frame columns are batch rather than biology (batch_index)
	#and subset the data frame to just those columns, in np.array form (dm)
	dummy = pd.get_dummies(adata.obs[batch_key+confounder_key],drop_first=False)
	if len(batch_key)>1:
		batch_index = np.logical_or.reduce(np.vstack([dummy.columns.str.startswith(x) for x in batch_key]))
	else:
		batch_index = np.vstack([dummy.columns.str.startswith(x) for x in batch_key])[0]
	dm = np.array(dummy)[:,batch_index]
	
	#compute how many genes at a time will be processed - aiming for chunksize total elements per
	chunkcount = np.ceil(chunksize/adata.shape[0])
	
	#make a Ridge with all the **kwargs passed if need be, and fit_intercept set to False
	#(as the data is centered). create holders for results
	LR = Ridge(fit_intercept=False, **kwargs)
	X_explained = []
	X_remain = []
	#loop over the gene space in chunkcount-sized chunks
	for ind in np.arange(0,adata.shape[1],chunkcount):
		#extract the expression and turn to dense if need be
		X_exp = adata.X[:,int(ind):int(ind+chunkcount)] # scaled data
		if scipy.sparse.issparse(X_exp):
			X_exp = X_exp.todense()
		#fit the ridge regression model, compute the expression explained by the technical 
		#effect, and the remaining residual
		LR.fit(dummy,X_exp)	
		X_explained.append(dm.dot(LR.coef_[:,batch_index].T))
		X_remain.append(X_exp - X_explained[-1])
	
	#collapse the chunked outputs and store them in the object
	X_explained = np.hstack(X_explained)
	X_remain = np.hstack(X_remain)
	adata.X = X_remain
	adata.layers['X_explained'] = X_explained
	logg.info('	finished', time=start,
		deep=('`.X` now features regression residuals\n'
		'	`.layers[\'X_explained\']` stores the expression explained by the technical effect'))
	return adata if copy else None

def extract_cell_connectivity(adata, cell, key='extracted_cell_connectivity'):
	'''
	Helper post-processing function that extracts a single cell's connectivity and stores
	it in ``adata.obs``, ready for plotting. Connectivities range from 0 to 1, the higher
	the connectivity the closer the cells are in the neighbour graph. Cells with a
	connectivity of 0 are unconnected in the graph.

	Input
	-----
	adata : ``AnnData``
		After having BBKNN ran on it.
	cell : ``str``
		The name of the cell to extract the connectivities for.
	key : ``str``, optional (default "extracted_cell_connectivity")
		What name to store the connectivities under in ``adata.obs``.
	'''
	if cell not in adata.obs_names:
		ValueError('The specified cell is not present in the object.')
	index = np.arange(len(adata.obs_names))[adata.obs_names==cell][0]
	adata.obs[key] = np.asarray(adata.uns['neighbors']['connectivities'][index,:].todense())[0]
