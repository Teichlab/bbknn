import numpy as np
import multiprocessing as mp
import sys
from scanpy import settings
from scanpy import logging as logg
from annoy import AnnoyIndex
from scipy.spatial import cKDTree
from sklearn.neighbors import KDTree
from scanpy.neighbors import compute_connectivities_umap

def create_tree(data,approx,metric):
	'''
	Create a cKDTree/KDTree/annoy index for nearest neighbour lookup. All undescribed input
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
		ckd.build(10)
	elif metric == 'euclidean':
		ckd = cKDTree(data)
	else:
		ckd = KDTree(data,metric=metric)
	return ckd

def query_tree(data,ckd,neighbors_within_batch,approx,metric,n_jobs):
	'''
	Query the cKDTree/KDTree/annoy index with PCA coordinates from a batch. All undescribed input
	as in ``bbknn.bbknn()``. Returns a tuple of distances and indices of neighbours for each cell
	in the batch.
	
	Input
	-----
	data : ``numpy.array``
		PCA coordinates of a batch's cells to query.
	ckd : cKDTree/KDTree/annoy index
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
		ckdout = ckd.query(x=data, k=neighbors_within_batch, n_jobs=n_jobs)
	else:
		ckdout = ckd.query(data, k=neighbors_within_batch)
	return ckdout

def get_graph(pca,batch_list,neighbors_within_batch,n_pcs,approx,metric,n_jobs):
	'''
	Identify the KNN structure to be used in graph construction. All input as in ``bbknn.bbknn()``
	and ``bbknn.bbknn_pca_matrix()``. Returns a tuple of distances and indices of neighbours for
	each cell.
	'''
	#get a list of all our batches
	batches = np.unique(batch_list)
	#create the output matrices, with the indices as integers and distances as floats
	knn_distances = np.zeros((pca.shape[0],neighbors_within_batch*len(batches)))
	knn_indices = np.copy(knn_distances).astype(int)
	#find the knns using cKDTree/KDTree/annoy
	#need to compare each batch against each batch (including itself)
	for to_ind in range(len(batches)):
		#this is the batch that will be used as the neighbour pool
		#create a boolean mask identifying the cells within this batch
		#and then get the corresponding row numbers for later use
		batch_to = batches[to_ind]
		mask_to = batch_list == batch_to
		ind_to = np.arange(len(batch_list))[mask_to]
		#create the cKDTree/KDTree/annoy, depending on approx/metric
		ckd = create_tree(data=pca[mask_to,:n_pcs],approx=approx,metric=metric)
		for from_ind in range(len(batches)):
			#this is the batch that will have its neighbours identified
			#repeat the mask/row number getting
			batch_from = batches[from_ind]
			mask_from = batch_list == batch_from
			ind_from = np.arange(len(batch_list))[mask_from]
			#fish the neighbours out, getting a (distances, indices) tuple back
			ckdout = query_tree(data=pca[mask_from,:n_pcs],ckd=ckd,
								neighbors_within_batch=neighbors_within_batch,
								approx=approx,metric=metric,n_jobs=n_jobs)
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

def scale_distances(knn_distances,batch_list,neighbors_within_batch):
	'''
	Scale the distances from disparate batches to be closer to the cell's batch of origin.
	Described in detail in ``bbknn.bbknn()``. All undescribed input as in ``bbknn.bbknn()``
	and ``bbknn.bbknn_pca_matrix()``. Returns a scaled distance array.
	
	Input
	-----
	knn_distances : ``numpy.array``
		Array of computed neighbour distances for each cell.
	'''
	#get a list of all our batches
	batches = np.unique(batch_list)
	for i in range(len(batches)):
		#where are our same-batch neighbours?
		inds = np.arange(len(batch_list))[batch_list == batches[i]]
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
	return knn_distances

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

def bbknn(adata, batch_key='batch', neighbors_within_batch=3, n_pcs=50, trim=None, 
		  scale_distance=False, approx=False, metric='euclidean', bandwidth=1, local_connectivity=1, 
		  n_jobs=None, save_knn=False, copy=False):
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
	approx : ``bool``, optional (default: ``False``)
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
	adata = adata.copy() if copy else adata
	#basic sanity checks to begin
	#is our batch key actually present in the object?
	if batch_key not in adata.obs:
		raise ValueError("Batch key '"+batch_key+"' not present in `adata.obs`.")
	#do we have a computed PCA? (the .dtype.fields is because of how adata.obsm is formatted)
	if 'X_pca' not in adata.obsm.dtype.fields:
		raise ValueError("`adata.obsm['X_pca']` doesn't exist. Run `sc.pp.pca` first.")
	#prepare bbknn_pca_matrix input
	pca = adata.obsm['X_pca']
	batch_list = adata.obs[batch_key].values
	#call BBKNN proper
	bbknn_out = bbknn_pca_matrix(pca=pca,batch_list=batch_list,neighbors_within_batch=neighbors_within_batch,
								 n_pcs=n_pcs,trim=trim,scale_distance=scale_distance,approx=approx,
								 metric=metric,bandwidth=bandwidth,local_connectivity=local_connectivity,
								 n_jobs=n_jobs,save_knn=save_knn)
	#optionally save knn_indices
	if save_knn:
		adata.uns['bbknn'] = bbknn_out[2]
	adata.uns['neighbors'] = {}
	adata.uns['neighbors']['params'] = {'n_neighbors': neighbors_within_batch*len(np.unique(batch_list)), 'method': 'umap'}
	adata.uns['neighbors']['distances'] = bbknn_out[0]
	adata.uns['neighbors']['connectivities'] = bbknn_out[1]
	logg.hint(
		'added to `.uns[\'neighbors\']`\n'
		'    \'distances\', weighted adjacency matrix\n'
		'    \'connectivities\', weighted adjacency matrix')
	return adata if copy else None

def bbknn_pca_matrix(pca, batch_list, neighbors_within_batch=3, n_pcs=50, trim=None, 
		  scale_distance=False, approx=False, metric='euclidean', bandwidth=1, local_connectivity=1, 
		  n_jobs=None, save_knn=False):
	'''
	Scanpy-independent BBKNN variant that runs on a PCA matrix and list of per-cell batch assignments instead of
	an AnnData object. Non-data-entry arguments behave the same way as ``bbknn.bbknn()``.
	Returns a ``(distances, connectivities)`` tuple, like what would have been stored in the AnnData object.
	The connectivities are the actual neighbourhood graph. If ``save_knn=True``, the tuple also
	includes the nearest neighbour indices for each cell as a third element.
	
	Input
	-----
	pca : ``numpy.array``
		PCA coordinates for each cell, with cells as rows.
	batch_list : ``numpy.array`` or ``list``
		A list of batch assignments for each cell.
	'''
	logg.info('computing batch balanced neighbors', r=True)
	#more basic sanity checks/processing
	#do we have the same number of cells in pca and batch_list?
	if pca.shape[0] != len(batch_list):
		raise ValueError("Different cell counts indicated by `pca.shape[0]` and `len(batch_list)`.")
	#find our core total
	if n_jobs == None:
		n_jobs = mp.cpu_count()
	#convert batch_list to np.array of strings for ease of mask making later
	batch_list = np.asarray([str(i) for i in batch_list])
	#obtain the batch balanced KNN graph
	knn_distances, knn_indices = get_graph(pca=pca,batch_list=batch_list,n_pcs=n_pcs,
										   approx=approx,metric=metric,n_jobs=n_jobs,
										   neighbors_within_batch=neighbors_within_batch)
	#distance scaling - move the minimum observed metric value for different batches
	#to the maximum metric value within the same batch as the cell originates from
	if scale_distance:
		knn_distances = scale_distances(knn_distances=knn_distances,batch_list=batch_list,
										neighbors_within_batch=neighbors_within_batch)
	#sort the neighbours so that they're actually in order from closest to furthest
	newidx = np.argsort(knn_distances,axis=1)
	knn_indices = knn_indices[np.arange(np.shape(knn_indices)[0])[:,np.newaxis],newidx]
	knn_distances = knn_distances[np.arange(np.shape(knn_distances)[0])[:,np.newaxis],newidx] 
	#this part of the processing is akin to scanpy.api.neighbors()
	dist, cnts = compute_connectivities_umap(knn_indices, knn_distances, knn_indices.shape[0], 
											 knn_indices.shape[1], bandwidth=bandwidth, 
											 local_connectivity=local_connectivity)
	#optional trimming
	if trim:
		cnts = trimming(cnts=cnts,trim=trim)
	logg.info('    finished', time=True, end=' ' if settings.verbosity > 2 else '\n')
	if save_knn:
		return (dist, cnts, knn_indices)
	return (dist, cnts)