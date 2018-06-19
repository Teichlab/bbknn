import numpy as np
import multiprocessing as mp
import sys
from scanpy import settings
from scanpy import logging as logg
from scipy.spatial import cKDTree
from sklearn.neighbors import KDTree
from scanpy.neighbors import compute_connectivities_umap

def bbknn(adata, batch_key='batch', neighbors_within_batch=3, metric='euclidean', n_pcs=50, scale_distance=False, n_jobs=None):
	'''
	Batch balanced KNN, identifying the top neighbours of each cell within each batch separately.
	For use in the scanpy workflow as an alternative to ``scanpi.api.pp.neighbors``.
	Similarly short run time (when using the default Euclidean metric) while correcting batch effect.
	
	Input
	-----
	adata : ``AnnData``
		Needs the PCA computed and stored in ``adata.obsm["X_pca"]``.
	batch_key : ``str``, optional (default "batch")
		``adata.obs`` column name discriminating between your batches.
	neighbors_within_batch : ``int``, optional (default 3)
		How many top neighbours to report for each batch; total number of neighbours 
		will be this number times the number of batches.
	metric : ``str`` or ``sklearn.neighbors.DistanceMetric``, optional (default "euclidean")
		What distance metric to use: "euclidean", "manhattan", "chebyshev", or 
		parameterised ``sklearn.neighbors.DistanceMetric`` for "minkowski", "wminkowski", 
		"seuclidean" or "mahalanobis".
		
		>>> from sklearn.neighbors import DistanceMetric
		>>> pass_this_as_metric = DistanceMetric.get_metric('minkowski',p=3)
	n_pcs : ``int``, optional (default 50)
		How many principal components to use in the analysis.
	scale_distance : ``Boolean``, optional (default False) 
		If True, lower the lowest across-batch distance to match the highest within-batch 
		neighbour\'s distance for each cell if needed.
	n_jobs : ``int`` or ``None``, optional (default ``None``)
		Parallelise neighbour identification when using an Euclidean distance metric, 
		if ``None`` use all cores. Does nothing with a different metric.
	'''
	logg.info('computing batch balanced neighbors', r=True)
	#basic sanity checks to begin
	#is our batch key actually present in the object?
	if batch_key not in adata.obs:
		raise ValueError("Batch key '"+batch_key+"' not present in `adata.obs`.")
	#do we have a computed PCA? (the .dtype.fields is because of how adata.obsm is formatted)
	if 'X_pca' not in adata.obsm.dtype.fields:
		raise ValueError("`adata.obsm['X_pca']` doesn't exist. Run `sc.pp.pca` first.")
	#find our core total
	if n_jobs == None:
		n_jobs = mp.cpu_count()
	#get a list of all our batches, which we'll use a few times later on
	batches = np.unique(adata.obs[batch_key])
	#create the output matrices, with the indices as integers and distances as floats
	knn_distances = np.zeros((adata.shape[0],neighbors_within_batch*len(batches)))
	knn_indices = np.copy(knn_distances).astype(int)
	#find the knns using cKDTree/KDTree
	#need to compare each batch against each batch (including itself)
	for to_ind in range(len(batches)):
		#this is the batch that will be used as the neighbour pool
		#create a boolean mask identifying the spots within adata that house cells within this batch
		#and then get the corresponding row numbers for later use
		batch_to = batches[to_ind]
		mask_to = adata.obs[batch_key] == batch_to
		ind_to = np.arange(adata.shape[0])[mask_to]
		#create the cKDTree/KDTree, depending on the metric
		if metric == 'euclidean':
			ckd = cKDTree(adata.obsm['X_pca'][mask_to,:n_pcs])
		else:
			ckd = KDTree(adata.obsm['X_pca'][mask_to,:n_pcs],metric=metric)
		for from_ind in range(len(batches)):
			#this is the batch that will have its neighbours identified
			#repeat the mask/row number getting
			batch_from = batches[from_ind]
			mask_from = adata.obs[batch_key] == batch_from
			ind_from = np.arange(adata.shape[0])[mask_from]
			#fish the neighbours out, getting a (distances, indices) tuple back
			if metric == 'euclidean':
				ckdout = ckd.query(x=adata.obsm['X_pca'][mask_from,:n_pcs], k=neighbors_within_batch, n_jobs=n_jobs)
			else:
				ckdout = ckd.query(adata.obsm['X_pca'][mask_from,:n_pcs], k=neighbors_within_batch)
			#the identified indices are relative to the subsetted PCA matrix
			#so we need to convert it back to the original adata row numbers
			for i in range(ckdout[1].shape[0]):
				for j in range(ckdout[1].shape[1]):
					ckdout[1][i,j] = ind_to[ckdout[1][i,j]]
			#save the results within the appropriate rows and columns of the structures
			col_range = np.arange(to_ind*neighbors_within_batch, (to_ind+1)*neighbors_within_batch)
			knn_indices[ind_from[:,None],col_range[None,:]] = ckdout[1]
			knn_distances[ind_from[:,None],col_range[None,:]] = ckdout[0]
	if scale_distance:
		#distance scaling - move the minimum observed metric value for different batches
		#to the maximum metric value within the same batch as the cell originates from
		for i in range(len(batches)):
			#where are our same-batch neighbours?
			inds = np.arange(adata.shape[0])[adata.obs[batch_key] == batches[i]]
			source_col_range = np.arange(i*neighbors_within_batch, (i+1)*neighbors_within_batch)
			for ind in inds:
				#the maximum observed metric value within the batch for this cell
				scale_value = np.max(knn_distances[ind,source_col_range])
				for j in range(len(batches)):
					#check against the minimum of the other batches, scale within batches if needed
					col_range = np.arange(j*neighbors_within_batch, (j+1)*neighbors_within_batch)
					if np.min(knn_distances[ind,col_range]) > scale_value:
						knn_distances[ind,col_range] = knn_distances[ind,col_range] + \
								scale_value - np.min(knn_distances[ind,col_range])
	#sort the neighbours so that they're actually in order from closest to furthest
	newidx = np.argsort(knn_distances,axis=1)
	knn_indices = knn_indices[np.arange(np.shape(knn_indices)[0])[:,np.newaxis],newidx]
	knn_distances = knn_distances[np.arange(np.shape(knn_distances)[0])[:,np.newaxis],newidx]
	#the rest of the processing is akin to scanpy.api.neighbors()
	dist, cnts = compute_connectivities_umap(knn_indices, knn_distances, knn_indices.shape[0], knn_indices.shape[1])
	adata.uns['neighbors'] = {}
	adata.uns['neighbors']['params'] = {'n_neighbors': knn_indices.shape[1], 'method': 'umap'}
	adata.uns['neighbors']['distances'] = dist
	adata.uns['neighbors']['connectivities'] = cnts
	logg.info('	finished', time=True, end=' ' if settings.verbosity > 2 else '\n')
	logg.hint(
		'added to `.uns[\'neighbors\']`\n'
		'    \'distances\', weighted adjacency matrix\n'
		'    \'connectivities\', weighted adjacency matrix')