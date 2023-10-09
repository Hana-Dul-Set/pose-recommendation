import random

from pyclustering.cluster.kmedians import kmedians
from pyclustering.utils.metric import type_metric, distance_metric

class KMediansClustering:
    def __init__(self, samples, metric_fn, K):
        self.samples = samples
        self.metric = distance_metric(type_metric.USER_DEFINED, func = metric_fn)
        self.K = K
        initial_centers = random.sample([i for i in range(len(samples))], K)
        self.instance = kmedians(samples, initial_centers, metric = self.metric)
        
    def process(self):
        self.instance.process()
    
    def get_clusters(self):
        self.instance.get_clusters()
    
    def get_medoid_indicies(self):
        self.instance.get_medians()
