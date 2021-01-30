import pandas as pd
import numpy as np
import scipy
import sys
from annoy import AnnoyIndex
from packaging import version
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from umap.umap_ import fuzzy_simplicial_set
from sklearn.neighbors import KDTree
from sklearn.neighbors import DistanceMetric
from sklearn.linear_model import Ridge
try:
	from scanpy import logging as logg
except ImportError:
	pass
try:
	import anndata
except ImportError:
	pass
try:
	import faiss
except ImportError:
	pass

def get_sparse_matrix_from_indices_distances_umap(knn_indices, knn_dists, n_obs, n_neighbors):
	'''
	Copied out of scanpy.neighbors
	'''
	rows = np.zeros((n_obs * n_neighbors), dtype=np.int64)
	cols = np.zeros((n_obs * n_neighbors), dtype=np.int64)
	vals = np.zeros((n_obs * n_neighbors), dtype=np.float64)

	for i in range(knn_indices.shape[0]):
		for j in range(n_neighbors):
			if knn_indices[i, j] == -1:
				continue  # We didn't get the full knn for i
			if knn_indices[i, j] == i:
				val = 0.0
			else:
				val = knn_dists[i, j]

			rows[i * n_neighbors + j] = i
			cols[i * n_neighbors + j] = knn_indices[i, j]
			vals[i * n_neighbors + j] = val

	result = coo_matrix((vals, (rows, cols)),
									  shape=(n_obs, n_obs))
	result.eliminate_zeros()
	return result.tocsr()

def compute_connectivities_umap(knn_indices, knn_dists,
		n_obs, n_neighbors, set_op_mix_ratio=1.0,
		local_connectivity=1.0):
	'''
	Copied out of scanpy.neighbors

	This is from umap.fuzzy_simplicial_set [McInnes18]_.
	Given a set of data X, a neighborhood size, and a measure of distance
	compute the fuzzy simplicial set (here represented as a fuzzy graph in
	the form of a sparse matrix) associated to the data. This is done by
	locally approximating geodesic distance at each point, creating a fuzzy
	simplicial set for each such point, and then combining all the local
	fuzzy simplicial sets into a global one via a fuzzy union.
	'''
	X = coo_matrix(([], ([], [])), shape=(n_obs, 1))
	connectivities = fuzzy_simplicial_set(X, n_neighbors, None, None,
										  knn_indices=knn_indices, knn_dists=knn_dists,
										  set_op_mix_ratio=set_op_mix_ratio,
										  local_connectivity=local_connectivity)
	if isinstance(connectivities, tuple):
		# In umap-learn 0.4, this returns (result, sigmas, rhos)
		connectivities = connectivities[0]
	distances = get_sparse_matrix_from_indices_distances_umap(knn_indices, knn_dists, n_obs, n_neighbors)
, n_neighbors)

	return distances, connectivities.tocsr()

def create_tree(data,approx,metric,use_faiss,n_trees):
	'''
	Create a faiss/cKDTree/KDTree/annoy index for nearest neighbour lookup. All undescribed input
	as in ``bbknn.bbknn()``. Returns the resulting index.

	Input
	-----
	data : ``numppy.array``
		PCA coordinates of a batch's cells to index.
	'''
	if approx:
		ckd = AnnoyIndex(data.shape[1],metric=metric)
		for i in np.arange(data.shape[0]):
			ckd.add_item(i,data[i,:])
		ckd.build(n_trees)
	elif metric == 'euclidean':
		if 'faiss' in sys.modules and use_faiss:
			ckd = faiss.IndexFlatL2(data.shape[1])
			ckd.add(data)
		else:
			ckd = cKDTree(data)
	else:
		ckd = KDTree(data,metric=metric)
	return ckd

def query_tree(data,ckd,neighbors_within_batch,approx,metric,use_faiss):
	'''
	Query the faiss/cKDTree/KDTree/annoy index with PCA coordinates from a batch. All undescribed input
	as in ``bbknn.bbknn()``. Returns a tuple of distances and indices of neighbours for each cell
	in the batch.

	Input
	-----
	data : ``numpy.array``
		PCA coordinates of a batch's cells to query.
	ckd : faiss/cKDTree/KDTree/annoy index
	'''
	if approx:
		ckdo_ind = []
		ckdo_dist = []
		for i in np.arange(data.shape[0]):
			holder = ckd.get_nns_by_vector(data[i,:],neighbors_within_batch,include_distances=True)
			ckdo_ind.append(holder[0])
			ckdo_dist.append(holder[1])
		ckdout = (np.asarray(ckdo_dist),np.asarray(ckdo_ind))
	elif metric == 'euclidean':
		if 'faiss' in sys.modules and use_faiss:
			D, I = ckd.search(data, neighbors_within_batch)
			#sometimes this turns up marginally negative values, just set those to zero
			D[D<0] = 0
			#the distance returned by faiss needs to be square rooted to be actual euclidean
			ckdout = (np.sqrt(D), I)
		else:
			ckdout = ckd.query(x=data, k=neighbors_within_batch, n_jobs=-1)
	else:
		ckdout = ckd.query(data, k=neighbors_within_batch)
	return ckdout

def get_graph(pca,batch_list,neighbors_within_batch,n_pcs,approx,metric,use_faiss,n_trees):
	'''
	Identify the KNN structure to be used in graph construction. All input as in ``bbknn.bbknn()``
	and ``bbknn.bbknn_pca_matrix()``. Returns a tuple of distances and indices of neighbours for
	each cell.
	'''
	#get a list of all our batches
	batches = np.unique(batch_list)
	#in case we're gonna be faissing, turn the data to float32
	if metric=='euclidean' and not approx and 'faiss' in sys.modules and use_faiss:
		pca = pca.astype('float32')
	#create the output matrices, with the indices as integers and distances as floats
	knn_distances = np.zeros((pca.shape[0],neighbors_within_batch*len(batches)))
	knn_indices = np.copy(knn_distances).astype(int)
	#find the knns using faiss/cKDTree/KDTree/annoy
	#need to compare each batch against each batch (including itself)
	for to_ind in range(len(batches)):
		#this is the batch that will be used as the neighbour pool
		#create a boolean mask identifying the cells within this batch
		#and then get the corresponding row numbers for later use
		batch_to = batches[to_ind]
		mask_to = batch_list == batch_to
		ind_to = np.arange(len(batch_list))[mask_to]
		#create the faiss/cKDTree/KDTree/annoy, depending on approx/metric
		ckd = create_tree(data=pca[mask_to,:n_pcs],approx=approx,metric=metric,
						  use_faiss=use_faiss,n_trees=n_trees)
		for from_ind in range(len(batches)):
			#this is the batch that will have its neighbours identified
			#repeat the mask/row number getting
			batch_from = batches[from_ind]
			mask_from = batch_list == batch_from
			ind_from = np.arange(len(batch_list))[mask_from]
			#fish the neighbours out, getting a (distances, indices) tuple back
			ckdout = query_tree(data=pca[mask_from,:n_pcs],ckd=ckd,
								neighbors_within_batch=neighbors_within_batch,
								approx=approx,metric=metric,use_faiss=use_faiss)
			#the identified indices are relative to the subsetted PCA matrix
			#so we need to convert it back to the original row numbers
			for i in range(ckdout[1].shape[0]):
				for j in range(ckdout[1].shape[1]):
					ckdout[1][i,j] = ind_to[ckdout[1][i,j]]
			#save the results within the appropriate rows and columns of the structures
			col_range = np.arange(to_ind*neighbors_within_batch, (to_ind+1)*neighbors_within_batch)
			knn_indices[ind_from[:,None],col_range[None,:]] = ckdout[1]
			knn_distances[ind_from[:,None],col_range[None,:]] = ckdout[0]
	return knn_distances, knn_indices

def trimming(cnts,trim):
	'''
	Trims the graph to the top connectivities for each cell. All undescribed input as in
	``bbknn.bbknn()``.

	Input
	-----
	cnts : ``CSR``
		Sparse matrix of processed connectivities to trim.
	'''
	vals = np.zeros(cnts.shape[0])
	for i in range(cnts.shape[0]):
		#Get the row slice, not a copy, only the non zero elements
		row_array = cnts.data[cnts.indptr[i]: cnts.indptr[i+1]]
		if row_array.shape[0] <= trim:
			continue
		#fish out the threshold value
		vals[i] = row_array[np.argsort(row_array)[-1*trim]]
	for iter in range(2):
		#filter rows, flip, filter columns using the same thresholds
		for i in range(cnts.shape[0]):
			#Get the row slice, not a copy, only the non zero elements
			row_array = cnts.data[cnts.indptr[i]: cnts.indptr[i+1]]
			if row_array.shape[0] <= trim:
				continue
			#apply cutoff
			row_array[row_array<vals[i]] = 0
		cnts.eliminate_zeros()
		cnts = cnts.T.tocsr()
	return cnts

def bbknn(adata, batch_key='batch', use_rep='X_pca', approx=True, metric='cosine', copy=False, **kwargs):
	'''
	Batch balanced KNN, altering the KNN procedure to identify each cell's top neighbours in
	each batch separately instead of the entire cell pool with no accounting for batch.
	Aligns batches in a quick and lightweight manner.
	For use in the scanpy workflow as an alternative to ``scanpi.api.pp.neighbors()``.

	Input
	-----
	adata : ``AnnData``
		Needs the PCA computed and stored in ``adata.obsm["X_pca"]``.
	batch_key : ``str``, optional (default: "batch")
		``adata.obs`` column name discriminating between your batches.
	neighbors_within_batch : ``int``, optional (default: 3)
		How many top neighbours to report for each batch; total number of neighbours
		will be this number times the number of batches.
	use_rep : ``str``, optional (default: "X_pca")
		The dimensionality reduction in ``.obsm`` to use for neighbour detection. Defaults to PCA.
	n_pcs : ``int``, optional (default: 50)
		How many dimensions (in case of PCA, principal components) to use in the analysis.
	trim : ``int`` or ``None``, optional (default: ``None``)
		Trim the neighbours of each cell to these many top connectivities. May help with
		population independence and improve the tidiness of clustering. The lower the value the
		more independent the individual populations, at the cost of more conserved batch effect.
		If ``None``, sets the parameter value automatically to 10 times the total number of
		neighbours for each cell. Set to 0 to skip.
	approx : ``bool``, optional (default: ``True``)
		If ``True``, use annoy's approximate neighbour finding. This results in a quicker run time
		for large datasets while also potentially increasing the degree of batch correction.
	n_trees : ``int``, optional (default: 10)
		Only used when ``approx=True``. The number of trees to construct in the annoy forest.
		More trees give higher precision when querying, at the cost of increased run time and
		resource intensity.
	use_faiss : ``bool``, optional (default: ``True``)
		If ``approx=False`` and the metric is "euclidean", use the faiss package to compute
		nearest neighbours if installed. This improves performance at a minor cost to numerical
		precision as faiss operates on float32.
	metric : ``str`` or ``sklearn.neighbors.DistanceMetric``, optional (default: "cosine")
		What distance metric to use. If using ``approx=True``, the options are "cosine",
		"euclidean", "manhattan" and "hamming". Otherwise, the options are "euclidean",
		a member of the ``sklearn.neighbors.KDTree.valid_metrics`` list, or parameterised
		``sklearn.neighbors.DistanceMetric`` `objects
		<https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html>`_:

		>>> from sklearn import neighbors
		>>> neighbors.KDTree.valid_metrics
		['p', 'chebyshev', 'cityblock', 'minkowski', 'infinity', 'l2', 'euclidean', 'manhattan', 'l1']
		>>> pass_this_as_metric = neighbors.DistanceMetric.get_metric('minkowski',p=3)
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
	#metric sanity checks
	if approx and metric not in ['euclidean', 'manhattan', 'hamming', 'cosine']:
		logg.warning('unrecognised metric for type of neighbor calculation, switching to cosine (')
		metric = 'cosine'
	elif not approx and not (metric=='euclidean' or isinstance(metric,DistanceMetric) or metric in KDTree.valid_metrics):
		logg.warning('unrecognised metric for type of neighbor calculation, switching to euclidean')
		metric = 'euclidean'
	#prepare bbknn_pca_matrix input
	pca = adata.obsm[use_rep]
	batch_list = adata.obs[batch_key].values
	#call BBKNN proper
	bbknn_out = bbknn_pca_matrix(pca=pca, batch_list=batch_list,
								 approx=approx, metric=metric, **kwargs)
	#store the parameters in .uns['neighbors']['params'], add use_rep and batch_key
	adata.uns['neighbors'] = {}
	adata.uns['neighbors']['params'] = bbknn_out[2]
	adata.uns['neighbors']['params']['use_rep'] = use_rep
	adata.uns['neighbors']['params']['bbknn']['batch_key'] = batch_key
	#store the graphs in .uns['neighbors'] or .obsp, conditional on anndata version
	if version.parse(str(anndata.__version__)) < version.parse('0.7.0'):
		adata.uns['neighbors']['distances'] = bbknn_out[0]
		adata.uns['neighbors']['connectivities'] = bbknn_out[1]
		logg.info('	finished', time=start,
			deep=('added to `.uns[\'neighbors\']`\n'
			'	\'distances\', distances for each pair of neighbors\n'
			'	\'connectivities\', weighted adjacency matrix'))
	else:
		adata.obsp['distances'] = bbknn_out[0]
		adata.obsp['connectivities'] = bbknn_out[1]
		adata.uns['neighbors']['distances_key'] = 'distances'
		adata.uns['neighbors']['connectivities_key'] = 'connectivities'
		logg.info('	finished', time=start,
			deep=('added to `.uns[\'neighbors\']`\n'
			'	`.obsp[\'distances\']`, distances for each pair of neighbors\n'
			'	`.obsp[\'connectivities\']`, weighted adjacency matrix'))
	return adata if copy else None

def bbknn_pca_matrix(pca, batch_list, neighbors_within_batch=3, n_pcs=50, trim=None,
		  approx=True, n_trees=10, use_faiss=True, metric='cosine',
		  set_op_mix_ratio=1, local_connectivity=1):
	'''
	Scanpy-independent BBKNN variant that runs on a PCA matrix and list of per-cell batch assignments instead of
	an AnnData object. Non-data-entry arguments behave the same way as ``bbknn.bbknn()``.
	Returns a ``(distances, connectivities, parameters)`` tuple, like what would have been stored in the AnnData object.
	The connectivities are the actual neighbourhood graph.

	Input
	-----
	pca : ``numpy.array``
		PCA (or other dimensionality reduction) coordinates for each cell, with cells as rows.
	batch_list : ``numpy.array`` or ``list``
		A list of batch assignments for each cell.
	'''
	#more basic sanity checks/processing
	#do we have the same number of cells in pca and batch_list?
	if pca.shape[0] != len(batch_list):
		raise ValueError("Different cell counts indicated by `pca.shape[0]` and `len(batch_list)`.")
	#convert batch_list to np.array of strings for ease of mask making later
	batch_list = np.asarray([str(i) for i in batch_list])
	#assert that all batches have at least neighbors_within_batch cells in there
	unique, counts = np.unique(batch_list, return_counts=True)
	if np.min(counts) < neighbors_within_batch:
		raise ValueError("Not all batches have at least `neighbors_within_batch` cells in them.")
	#metric sanity checks (duplicating the ones in bbknn(), but without scanpy logging)
	if approx and metric not in ['cosine', 'euclidean', 'manhattan', 'hamming']:
		print('unrecognised metric for type of neighbor calculation, switching to cosine')
		metric = 'cosine'
	elif not approx and not (metric=='euclidean' or isinstance(metric,DistanceMetric) or metric in KDTree.valid_metrics):
		print('unrecognised metric for type of neighbor calculation, switching to euclidean')
		metric = 'euclidean'
	#obtain the batch balanced KNN graph
	knn_distances, knn_indices = get_graph(pca=pca,batch_list=batch_list,n_pcs=n_pcs,n_trees=n_trees,
										   approx=approx,metric=metric,use_faiss=use_faiss,
										   neighbors_within_batch=neighbors_within_batch)
	#sort the neighbours so that they're actually in order from closest to furthest
	newidx = np.argsort(knn_distances,axis=1)
	knn_indices = knn_indices[np.arange(np.shape(knn_indices)[0])[:,np.newaxis],newidx]
	knn_distances = knn_distances[np.arange(np.shape(knn_distances)[0])[:,np.newaxis],newidx]
	#this part of the processing is akin to scanpy.api.neighbors()
	dist, cnts = compute_connectivities_umap(knn_indices, knn_distances, knn_indices.shape[0],
											 knn_indices.shape[1], set_op_mix_ratio=set_op_mix_ratio,
											 local_connectivity=local_connectivity)
	#trimming. compute default range if absent
	if trim is None:
		trim = 10 * knn_distances.shape[1]
	#skip trimming if set to 0, otherwise trim
	if trim > 0:
		cnts = trimming(cnts=cnts,trim=trim)
	#create a collated parameters dictionary
	#determine which neighbour computation was used, mirroring create_tree() logic
	if approx:
		computation='annoy'
	elif metric == 'euclidean':
		if 'faiss' in sys.modules and use_faiss:
			computation='faiss'
		else:
			computation='cKDTree'
	else:
		computation='KDTree'
	#we'll have a zero distance for our cell of origin, and nonzero for every other neighbour computed
	params = {'n_neighbors': len(dist[0,:].data)+1, 'method': 'umap', 
			  'metric': metric, 'n_pcs': n_pcs, 
			  'bbknn': {'trim': trim, 'computation': computation}}
	return (dist, cnts, params)

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
		X_exp = adata.X[:,np.int(ind):np.int(ind+chunkcount)] # scaled data
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
