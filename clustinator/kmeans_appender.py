'''
@author: Henning Schulz
'''

from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
import numpy as np
import time
from datetime import datetime
from _warnings import warn

from session_appender import SessionAppender

class KmeansAppender(SessionAppender):
    
    def __init__(self, prev_behavior_models, k, n_jobs=1, dimensions=None):
        self.k = k
        self.n_jobs = n_jobs
        self.dimensions = dimensions
        
        if prev_behavior_models:
            self._do_remap = True
        else:
            self._do_remap = False
    
    def _do_clustering(self, csr_matrix):
        if self.dimensions:
            print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), 'Reducing the sessions to', self.dimensions, 'dimensions...')
            reduced_matrix = TruncatedSVD(n_components=self.dimensions).fit_transform(csr_matrix)
        else:
            reduced_matrix = csr_matrix
        
        
        print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), 'Starting the clustering with k =', self.k, 'and n_jobs =', self.n_jobs, '...')
        labels = KMeans(n_clusters=self.k, n_jobs=self.n_jobs).fit(reduced_matrix).labels_
        unique, counts = np.unique(labels, return_counts=True)
        print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), 'Clustering done. Found the following clusters:', unique, 'with counts:', counts)
        
        return unique, counts, labels
    
    def append(self, csr_matrix):
        unique, counts, labels = self._do_clustering(csr_matrix)
        
        print("Calculating the cluster means...")
        cluster_means = self._calculate_cluster_means(csr_matrix, labels)
        print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), 'Mean calculation done.')
        
        if self._do_remap:
            warn('Remapping the clusters to previous ones is not implemented yet! Use the minimum-distance strategy instead.')
        
        self.cluster_means = cluster_means
        self.labels = labels
        self.cluster_mapping = None
        
        self.num_sessions = { str(mid) : count for mid, count in zip(unique, counts) }
