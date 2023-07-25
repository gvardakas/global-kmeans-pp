from abc import ABC, abstractmethod
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.utils.validation import check_array, check_is_fitted

class _BaseGlobalKMeans(BaseEstimator, ClusterMixin, TransformerMixin, ABC):
	"""Base class for Global K-Means, Global K-Means++ and future (or past) variants.

		Parameters
		----------
			n_clusters (int) : The number of clusters to form and the number of centroids to generate.
			tol (float) : Relative tolerance with regards to Frobenius norm of the difference in the cluster centers of two consecutive iterations to declare convergence.
			verbose (int) : Verbosity mode.

		Attributes
		----------
			n_iter_ (int) : The total number of KMeans iterations.
			cluster_centers_ (dict) : Dictionary to store cluster centers for each sub-problem k.
			labels_ (dict) : Dictionary to store cluster labels for each sub-problem k.
			inertia_ (dict) : Dictionary to store inertia values for each sub-problem k.
	"""
	def __init__(self, n_clusters, tol, verbose):
		self.n_clusters = n_clusters
		self.tol = tol
		self.verbose = verbose
		
		self.n_iter_ = 0
		self.cluster_centers_ = {}
		self.labels_ = {}
		self.inertia_ = {}

	@abstractmethod	
	def fit(self, X, y=None, sample_weight=None):
		"""Abstract method for fitting the model to the data.

		Parameters
		----------
			X (array-like) : Input data.
			y : Ignored
			sample_weight : Ignored

		Returns
		----------
			self: Fitted estimator.
		"""
		return self

	def predict(self, X=None):
		"""Predict cluster labels for data X with n_clusters clusters.

		Parameters
		----------
			X : Ignored. Models should already be fitted.

		Returns
		----------
			labels_ (array) : Cluster labels.
		"""
		return self.labels_[self.n_clusters]

	def fit_predict(self, X, y=None, sample_weight=None):
		"""Compute cluster centers and predict cluster index for each sample.

        Convenience method; equivalent to calling fit(X) followed by predict(X).


		Parameters
		----------
			X (array-like) : Input data.

		Returns
		----------
			labels_ (array) : Cluster labels.
		"""
		self.fit(X, y, sample_weight)
		return self.predict(X)


class GlobalKMeans(_BaseGlobalKMeans):
	"""Global K-Means clustering algorithm.

		Parameters:
			n_clusters (int) : The number of clusters to form and the number of centroids to generate.
			tol (float) : Relative tolerance with regards to Frobenius norm of the difference in the cluster centers of two consecutive iterations to declare convergence.
			verbose (int) : Verbosity mode.

		Attributes
		----------
			n_iter_ (int) : The total number of KMeans iterations.
			cluster_centers_ (dict) : Dictionary to store cluster centers for each sub-problem k.
			labels_ (dict) : Dictionary to store cluster labels for each sub-problem k.
			inertia_ (dict) : Dictionary to store inertia values for each sub-problem k.

	"""
	def __init__(self, n_clusters=8, tol=1e-4, verbose=0):
		super().__init__(
			n_clusters=n_clusters,
			tol=tol,
			verbose=verbose,
		)

	def fit(self, X, y=None, sample_weight=None):
		"""Compute the global k-means clustering.

		Parameters
		----------
			X (array-like) : Input data.
			y : Ignored
			sample_weight : Ignored

		Returns
		----------
			self : Fitted estimator.
		"""
		check_array(X)
		kmeans = KMeans(n_clusters=1, init='random', n_init=1, tol=self.tol).fit(X)
		self.cluster_centers_[1] = kmeans.cluster_centers_
		self.labels_[1] = kmeans.labels_
		self.inertia_[1] = kmeans.inertia_
		
		for k in range(2, self.n_clusters+1):
			if self.verbose > 0: 
				print(f'Solving {k}-means')
						
			self.inertia_[k] = float('inf')
			for i, xi in enumerate(X): # TODO parallel
				current_centroids = np.vstack((self.cluster_centers_[k-1], xi))
				kmeans = KMeans(n_clusters=k, init=current_centroids, n_init=1, tol=self.tol)
				kmeans = kmeans.fit(X)
				self.n_iter_ += kmeans.n_iter_
				
				if kmeans.inertia_ < self.inertia_[k]:
					self.cluster_centers_[k] = kmeans.cluster_centers_
					self.labels_[k] = kmeans.labels_
					self.inertia_[k] = kmeans.inertia_

		return self

class GlobalKMeansPP(_BaseGlobalKMeans):
	"""Global K-Means++ clustering algorithm.

		Parameters
		----------
			n_clusters (int) : The number of clusters to form and the number of centroids to generate.
			n_candidates (int) : The number of centroid candidates to examine.
			sampling (str): The sampling method utilized for the centroid candidates ('batch' or 'sequential').
			tol (float) : Relative tolerance with regards to Frobenius norm of the difference in the cluster centers of two consecutive iterations to declare convergence.
			verbose (int) : Verbosity mode.
				
		Attributes
		----------
			n_iter_ (int) : The total number of KMeans iterations.
			cluster_centers_ (dict) : Dictionary to store cluster centers for each sub-problem k.
			labels_ (dict) : Dictionary to store cluster labels for each sub-problem k.
			inertia_ (dict) : Dictionary to store inertia values for each sub-problem k.
			cluster_distance_space_(dict) : Dictionary to store the distance of each data point to the nearest centroid.
	"""
	def __init__(self, n_clusters=8, n_candidates=25, sampling='batch', tol=1e-4, verbose=0):
		super().__init__(
			n_clusters=n_clusters,
			tol=tol,
			verbose=verbose,
		)
		self.n_candidates = n_candidates
		self.sampling = sampling
		
		self.cluster_distance_space_ = dict()
	
	def fit(self, X, y=None, sample_weight=None):
		"""Compute the global k-means++ clustering.

		Parameters
		----------
			X (array-like) : Input data.
			y : Ignored
			sample_weight : Ignored

		Returns
		----------
			self : Fitted estimator.
		"""
		check_array(X)
		kmeans = KMeans(n_clusters=1, init='random', n_init=1, tol=self.tol).fit(X)
		self.n_data = X.shape[0]
		self.cluster_centers_[1] = kmeans.cluster_centers_
		self.labels_[1] = kmeans.labels_
		self.inertia_[1] = kmeans.inertia_
		self.cluster_distance_space_[1] = kmeans.transform(X).min(axis=1)
		
		for k in range(2, self.n_clusters+1):
			if self.verbose > 0: 
				print(f'Solving {k}-means')
				
			centroid_candidates = self._sampling(X, self.cluster_distance_space_[k-1])
			
			self.inertia_[k] = float('inf')
			for i, xi in enumerate(centroid_candidates): # TODO parallel
				current_centroids = np.vstack((self.cluster_centers_[k-1], xi))
				kmeans = KMeans(n_clusters=k, init=current_centroids, n_init=1, tol=self.tol)
				kmeans = kmeans.fit(X)
				self.n_iter_ += kmeans.n_iter_

				if kmeans.inertia_ < self.inertia_[k]:
					self.cluster_centers_[k] = kmeans.cluster_centers_
					self.labels_[k] = kmeans.labels_
					self.inertia_[k] = kmeans.inertia_
					self.cluster_distance_space_[k] = kmeans.transform(X).min(axis=1)
		
		return self
	
	def _sampling(self, X, cluster_distance_space):
		if self.sampling == 'batch':
			return self._kmeans_pp_batch(X, cluster_distance_space)
		elif self.sampling == 'sequential':
			return self._kmeans_pp_sequential(X, cluster_distance_space)
		else:
			raise ValueError("Wrong sampling method! options = ['batch', 'sequential']")
	
	def _kmeans_pp_batch(self, X, cluster_distance_space):
		cluster_distance_space = np.power(cluster_distance_space, 2).flatten()
		sum_distance = np.sum(cluster_distance_space)
		selection_prob = cluster_distance_space / sum_distance
		selected_indexes = np.random.choice(self.n_data, size=self.n_candidates, p=selection_prob, replace=False) 
		return X[selected_indexes]
	
	def _kmeans_pp_sequential(self, X, cluster_distance_space):
		selected_indexes = np.zeros(shape=(self.n_candidates), dtype=np.int32)       
		cluster_distance_space = cluster_distance_space.flatten()
		for i in range(self.n_candidates):
			cluster_distance_space_squared = np.power(cluster_distance_space, 2)
			sum_distance = np.sum(cluster_distance_space_squared)
			selection_prob = cluster_distance_space_squared / sum_distance
			selected_indexes[i] = np.random.choice(self.n_data, size=1, p=selection_prob, replace=False)
			
			candidate = X[selected_indexes[i]]
			candidate = np.reshape(candidate, newshape=(1, candidate.shape[0]))
			candidate_data_distances = pairwise_distances(candidate, X).flatten()

			# Update probability distribution
			for i, _ in enumerate(cluster_distance_space):
				if candidate_data_distances[i] < cluster_distance_space[i]:
					cluster_distance_space[i] = candidate_data_distances[i]
		return X[selected_indexes]