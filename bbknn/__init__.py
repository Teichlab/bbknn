import numpy as np
import multiprocessing as mp
import sys
from scanpy import settings
from scanpy import logging as logg
from annoy import AnnoyIndex
from scipy.spatial import cKDTree
from sklearn.neighbors import KDTree
from scanpy.neighbors import compute_connectivities_umap

def bbknn(adata, batch_key='batch', neighbors_within_batch=3, n_pcs=50, trim=None, scale_distance=False, approx=False, metric='euclidean', bandwidth=1, local_connectivity=1, n_jobs=None, save_knn=False, copy=False):
	'''
	Batch balanced KNN, identifying the top neighbours of each cell within each batch separately.
	For use in the scanpy workflow as an alternative to ``scanpi.api.pp.neighbors``.
	Similarly short run time (when using the default Euclidean metric) while aligning batches.
	
	Input
	-----
	adata : ``AnnData``
		Needs the PCA computed and stored in ``adata.obsm["X_pca"]``.
	batch_key : ``str``, optional (default: "batch")
		``adata.obs`` column name discriminating between your batches.
	neighbors_within_batch : ``int``, optional (default: 3)
		How many top neighbours to report for each batch; total number of neighbours 
		will be this number times the number of batches.
	n_pcs : ``int``, optional (default: 50)
		How many principal components to use in the analysis.
	trim : ``int`` or ``None``, optional (default: ``None``)
		If not ``None``, trim the neighbours of each cell to these many top connectivities.
		May help with population independence and improve the tidiness of clustering.
	scale_distance : ``bool``, optional (default: ``False``) 
		If ``True``, optionally lower the across-batch distances on a per-cell, per-batch basis to make
		the closest neighbour be closer to the furthest within-batch neighbour. 
		May help smooth out very severe batch effects with a risk of overly 
		connecting the cells. The exact algorithm is as follows:
		
		.. code-block:: python
		
			if min(corrected_batch) > max(original_batch):
				corrected_batch += max(original_batch) - min(corrected_batch) + np.std(corrected_batch)
	approx :  ``bool``, optional (default: ``False``)
		If ``True``, use annoy's approximate neighbour finding. This results in a quicker run time 
		for large datasets at a risk of loss of independence of some of the populations. It should
		be noted that annoy's default metric of choice is "angular", which BBKNN overrides to
		"euclidean" from its own default metric setting.
	metric : ``str`` or ``sklearn.neighbors.DistanceMetric``, optional (default: "euclidean")
		What distance metric to use. If using ``approx=True``, the options are "euclidean",
		"angular", "manhattan" and "hamming". Otherwise, the options are "euclidean", 
		"manhattan", "chebyshev", or parameterised ``sklearn.neighbors.DistanceMetric`` 
		for "minkowski", "wminkowski", "seuclidean" or "mahalanobis".
		
		>>> from sklearn.neighbors import DistanceMetric
		>>> pass_this_as_metric = DistanceMetric.get_metric('minkowski',p=3)
	bandwidth : ``float``, optional (default: 1)
		``scanpy.neighbors.compute_connectivities_umap`` parameter, higher values result in a
		gentler slope of the connectivities exponentials (i.e. larger connectivity values being returned)
	local_connectivity : ``int``, optional (default: 1)
		``scanpy.neighbors.compute_connectivities_umap`` parameter, how many nearest neighbors of
		each cell are assumed to be fully connected (and given a connectivity value of 1)
	n_jobs : ``int`` or ``None``, optional (default: ``None``)
		Parallelise neighbour identification when using an Euclidean distance metric, 
		if ``None`` use all cores. Does nothing with a different metric.
	save_knn : ``bool``, optional (default: ``False``)
		If ``True``, save the indices of the nearest neighbours for each cell in ``adata.uns['bbknn']``.
	copy : ``bool``, optional (default: ``False``)
		If ``True``, return a copy instead of writing to the supplied adata.
	'''
	logg.info('computing batch balanced neighbors', r=True)
	adata = adata.copy() if copy else adata
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
		if approx:
			ckd = AnnoyIndex(n_pcs,metric=metric)
			data = adata.obsm['X_pca'][mask_to,:n_pcs]
			for i in np.arange(data.shape[0]):
				ckd.add_item(i,data[i,:])
			ckd.build(10)
		elif metric == 'euclidean':
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
			if approx:
				ckdo_ind = []
				ckdo_dist = []
				data = adata.obsm['X_pca'][mask_from,:n_pcs]
				for i in np.arange(data.shape[0]):
					holder = ckd.get_nns_by_vector(data[i,:],neighbors_within_batch,include_distances=True)
					ckdo_ind.append(holder[0])
					ckdo_dist.append(holder[1])
				ckdout = (np.asarray(ckdo_dist),np.asarray(ckdo_ind))
			elif metric == 'euclidean':
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
								scale_value - np.min(knn_distances[ind,col_range]) + \
								np.std(knn_distances[ind,col_range])
	#sort the neighbours so that they're actually in order from closest to furthest
	newidx = np.argsort(knn_distances,axis=1)
	knn_indices = knn_indices[np.arange(np.shape(knn_indices)[0])[:,np.newaxis],newidx]
	knn_distances = knn_distances[np.arange(np.shape(knn_distances)[0])[:,np.newaxis],newidx] 
	#optionally save knn_indices
	if save_knn:
		adata.uns['bbknn'] = knn_indices
	#this part of the processing is akin to scanpy.api.neighbors()
	dist, cnts = compute_connectivities_umap(knn_indices, knn_distances, knn_indices.shape[0], knn_indices.shape[1], bandwidth=bandwidth, local_connectivity=local_connectivity)
	#optional trimming
	if trim:
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
	adata.uns['neighbors'] = {}
	adata.uns['neighbors']['params'] = {'n_neighbors': knn_indices.shape[1], 'method': 'umap'}
	adata.uns['neighbors']['distances'] = dist
	adata.uns['neighbors']['connectivities'] = cnts
	logg.info('    finished', time=True, end=' ' if settings.verbosity > 2 else '\n')
	logg.hint(
		'added to `.uns[\'neighbors\']`\n'
		'    \'distances\', weighted adjacency matrix\n'
		'    \'connectivities\', weighted adjacency matrix')
	return adata if copy else None
