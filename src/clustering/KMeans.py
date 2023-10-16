import random

from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.utils.metric import type_metric, distance_metric

class KMeansClustering:
    def __init__(self, samples, metric_fn, K):
        self.samples = samples
        self.metric = distance_metric(type_metric.USER_DEFINED, func = metric_fn)
        self.K = K
        #initial_centers = random.sample([i for i in range(len(samples))], K)
        #initial_centers = [samples[i] for i in initial_centers]
        initial_centers = kmeans_plusplus_initializer(samples, K).initialize()
        self.instance = kmeans(samples, initial_centers, metric = self.metric)
        
    def process(self):
        self.instance.process()
    
    def get_labels(self):
        labels = [-1 for i in range(self.K)] 
        for clusterid, cluster in enumerate(self.instance.get_clusters()):
            for x in cluster:
                labels[x] = clusterid
        return labels
    
    def get_centers(self):
        return self.instance.get_centers()

    def get_total_error(self):
        return self.instance.get_total_wce()