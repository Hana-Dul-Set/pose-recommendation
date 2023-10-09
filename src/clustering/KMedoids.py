import random

from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.utils.metric import type_metric, distance_metric

class KMedoidsClustering:
    def __init__(self, samples, metric_fn, K):
        self.samples = samples
        self.metric = distance_metric(type_metric.USER_DEFINED, func = metric_fn)
        self.K = K
        initial_centers = random.sample([i for i in range(len(samples))], K)
        self.instance = kmedoids(samples, initial_centers, metric = self.metric)
        
    def process(self):
        self.instance.process()
    
    def get_clusters(self):
        return self.instance.get_clusters()
    
    def get_medoid_indicies(self):
        return self.instance.get_medoids()
    
