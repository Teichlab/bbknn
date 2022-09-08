import pandas as pd
import numpy as np
import scipy
import types
import sys
from annoy import AnnoyIndex
import pynndescent
from packaging import version
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from umap.umap_ import fuzzy_simplicial_set
from sklearn.neighbors import KDTree
from sklearn.neighbors import DistanceMetric
try:
	from scanpy import logging as logg
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

	return distances, connectivities.tocsr()

def create_tree(data, params):
	'''
	Create a faiss/cKDTree/KDTree/annoy/pynndescent index for nearest neighbour lookup. 
	All undescribed input as in ``bbknn.bbknn()``. Returns the resulting index.

	Input
	-----
	data : ``numpy.array``
		PCA coordinates of a batch's cells to index.
	params : ``dict``
		A dictionary of arguments used to call ``bbknn.matrix.bbknn()``, plus ['computation']
		storing the knn algorithm to use.
	'''
	if params['computation'] == 'annoy':
		ckd = AnnoyIndex(data.shape[1],metric=params['metric'])
		for i in np.arange(data.shape[0]):
			ckd.add_item(i,data[i,:])
		ckd.build(params['annoy_n_trees'])
	elif params['computation'] == 'pynndescent':
		ckd = pynndescent.NNDescent(data, metric=params['metric'], n_jobs=-1,
									n_neighbors=params['pynndescent_n_neighbors'], 
									random_state=params['pynndescent_random_state'])
		ckd.prepare()
	elif params['computation'] == 'faiss':
		ckd = faiss.IndexFlatL2(data.shape[1])
		ckd.add(data)
	elif params['computation'] == 'cKDTree':
		ckd = cKDTree(data)
	elif params['computation'] == 'KDTree':
		ckd = KDTree(data,metric=params['metric'])
	return ckd

def query_tree(data, ckd, params):
	'''
	Query the faiss/cKDTree/KDTree/annoy index with PCA coordinates from a batch. All undescribed input
	as in ``bbknn.bbknn()``. Returns a tuple of distances and indices of neighbours for each cell
	in the batch.

	Input
	-----
	data : ``numpy.array``
		PCA coordinates of a batch's cells to query.
	ckd : faiss/cKDTree/KDTree/annoy/pynndescent index
	params : ``dict``
		A dictionary of arguments used to call ``bbknn.matrix.bbknn()``, plus ['computation']
		storing the knn algorithm to use.
	'''
	if params['computation'] == 'annoy':
		ckdo_ind = []
		ckdo_dist = []
		for i in np.arange(data.shape[0]):
			holder = ckd.get_nns_by_vector(data[i,:],params['neighbors_within_batch'],include_distances=True)
			ckdo_ind.append(holder[0])
			ckdo_dist.append(holder[1])
		ckdout = (np.asarray(ckdo_dist),np.asarray(ckdo_ind))
	elif params['computation'] == 'pynndescent':
		ckdout = ckd.query(data, k=params['neighbors_within_batch'])
		ckdout = (ckdout[1], ckdout[0])
	elif params['computation'] == 'faiss':
		D, I = ckd.search(data, params['neighbors_within_batch'])
		#sometimes this turns up marginally negative values, just set those to zero
		D[D<0] = 0
		#the distance returned by faiss needs to be square rooted to be actual euclidean
		ckdout = (np.sqrt(D), I)
	elif params['computation'] == 'cKDTree':
		ckdout = ckd.query(x=data, k=params['neighbors_within_batch'], n_jobs=-1)
	elif params['computation'] == 'KDTree':
		ckdout = ckd.query(data, k=params['neighbors_within_batch'])
	return ckdout

def get_graph(pca, batch_list, params):
	'''
	Identify the KNN structure to be used in graph construction. All input as in ``bbknn.bbknn()``
	and ``bbknn.matrix.bbknn()``. Returns a tuple of distances and indices of neighbours for
	each cell.
	
	Input
	-----
	params : ``dict``
		A dictionary of arguments used to call ``bbknn.matrix.bbknn()``, plus ['computation']
		storing the knn algorithm to use.
	'''
	#get a list of all our batches
	batches = np.unique(batch_list)
	#in case we're gonna be faissing, turn the data to float32
	if params['computation'] == 'faiss':
		pca = pca.astype('float32')
	#create the output matrices, with the indices as integers and distances as floats
	knn_distances = np.zeros((pca.shape[0],params['neighbors_within_batch']*len(batches)))
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
		ckd = create_tree(data=pca[mask_to,:params['n_pcs']], params=params)
		ckdout = query_tree(data=pca[:,:params['n_pcs']], ckd=ckd, params=params)
		col_range = np.arange(to_ind*params['neighbors_within_batch'], (to_ind+1)*params['neighbors_within_batch'])
		knn_indices[:,col_range] = ind_to[ckdout[1]]
		knn_distances[:,col_range] = ckdout[0]
	return knn_distances, knn_indices

def check_knn_metric(params, counts, scanpy_logging=False):
	'''
	Checks if the provided metric can be used with the implied KNN algorithm. Returns parameters
	with the metric altered and the KNN algorithm stated outright in params['computation'].
	
	Input
	-----
	params : ``dict``
		A dictionary of arguments used to call ``bbknn.matrix.bbknn()``
	counts : ``np.array``
		The number of cells in each batch
	scanpy_logging : ``bool``, optional (default: ``False``)
		Whether to use scanpy logging to print updates rather than a ``print()``
	'''
	#take note if we end up going back to Euclidean
	swapped = False
	if params['approx']:
		#we're approximate
		if params['use_annoy']:
			params['computation'] = 'annoy'
			if params['metric'] not in ['angular', 'euclidean', 'manhattan', 'hamming']:
				swapped = True
				params['metric'] = 'euclidean'
		else:
			params['computation'] = 'pynndescent'
			#pynndescent wants at least 11 cells per batch, from testing
			if np.min(counts) < 11:
				raise ValueError("Not all batches have at least 11 cells in them - required by pynndescent.")
			#metric needs to be a function or in the named list
			if not (params['metric'] in pynndescent.distances.named_distances or
					isinstance(params['metric'], types.FunctionType)):
				swapped = True
				params['metric'] = 'euclidean'
	else:
		#we're not approximate
		#metric needs to either be a DistanceMetric object or fall in the KDTree name list
		if not (params['metric']=='euclidean' or 
				isinstance(params['metric'], DistanceMetric) or 
				params['metric'] in KDTree.valid_metrics):
			swapped = True
			params['metric'] = 'euclidean'
		if params['metric']=='euclidean':
			if 'faiss' in sys.modules and params['use_faiss']:
				params['computation']='faiss'
			else:
				params['computation']='cKDTree'
		else:
			params['computation']='KDTree'
	if swapped:
		#need to let the user know we swapped the metric
		if scanpy_logging:
			logg.warning('unrecognised metric for type of neighbor calculation, switching to euclidean')
		else:
			print('unrecognised metric for type of neighbor calculation, switching to euclidean')
	return params

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
			#apply cutoff
			row_array[row_array<vals[i]] = 0
		cnts.eliminate_zeros()
		cnts = cnts.T.tocsr()
	return cnts

def bbknn(pca, batch_list, neighbors_within_batch=3, n_pcs=50, trim=None,
		  approx=True, annoy_n_trees=10, pynndescent_n_neighbors=30, 
		  pynndescent_random_state=0, use_annoy=True, use_faiss=True, 
		  metric='euclidean', set_op_mix_ratio=1, local_connectivity=1):
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
	#catch all arguments for easy passing to subsequent functions
	params = locals()
	del params['pca']
	del params['batch_list']
	#more basic sanity checks/processing
	#do we have the same number of cells in pca and batch_list?
	if pca.shape[0] != len(batch_list):
		raise ValueError("Different cell counts indicated by `pca.shape[0]` and `len(batch_list)`.")
	#convert batch_list to np.array of strings for ease of mask making later
	batch_list = np.asarray([str(i) for i in batch_list])
	#assert that all batches have at least neighbors_within_batch cells in there
	unique, counts = np.unique(batch_list, return_counts=True)
	if np.min(counts) < params['neighbors_within_batch']:
		raise ValueError("Not all batches have at least `neighbors_within_batch` cells in them.")
	#so what knn algorithm will be using? sanity check the metrics while at it
	params = check_knn_metric(params, counts)
	#obtain the batch balanced KNN graph
	knn_distances, knn_indices = get_graph(pca=pca,batch_list=batch_list,params=params)
	#sort the neighbours so that they're actually in order from closest to furthest
	newidx = np.argsort(knn_distances,axis=1)
	knn_indices = knn_indices[np.arange(np.shape(knn_indices)[0])[:,np.newaxis],newidx]
	knn_distances = knn_distances[np.arange(np.shape(knn_distances)[0])[:,np.newaxis],newidx]
	#this part of the processing is akin to scanpy.api.neighbors()
	dist, cnts = compute_connectivities_umap(knn_indices, knn_distances, knn_indices.shape[0],
											 knn_indices.shape[1], set_op_mix_ratio=set_op_mix_ratio,
											 local_connectivity=local_connectivity)
	#trimming. compute default range if absent
	if params['trim'] is None:
		trim = 10 * knn_distances.shape[1]
	#skip trimming if set to 0, otherwise trim
	if trim > 0:
		cnts = trimming(cnts=cnts,trim=trim)
	#create a collated parameters dictionary
	#we'll have a zero distance for our cell of origin, and nonzero for every other neighbour computed
	params = {'n_neighbors': len(dist[0,:].data)+1, 'method': 'umap', 
			  'metric': params['metric'], 'n_pcs': params['n_pcs'], 
			  'bbknn': {'trim': params['trim'], 'computation': params['computation']}}
	return (dist, cnts, params)
