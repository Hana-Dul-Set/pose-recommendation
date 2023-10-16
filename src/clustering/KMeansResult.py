import json
from tqdm import tqdm

class KMeansResult:
    def __init__(self, load_path = ""):
        if load_path != "":
            self.load(load_path)
    
    def set_result(self, K, loss, labels, centroids, names):
        self.K = K
        self.loss = loss
        self.labels = labels
        self.centroids = centroids
        self.names = names

    def get_loss(self):
        return self.loss
    
    def get_names(self):
        return self.names
    
    def get_K(self):
        return self.K
    
    def get_centroids(self):
        return self.centroids
    
    def get_centroid_images(self, preprocessed_datas):
        indicies = ClosestImages(self.centroids, preprocessed_datas).get_closest_indicies()
        return [self.names[idx] for idx in indicies]
    
    def get_groups(self):
        groups = [[] for i in range(self.K)]
        for i, name in enumerate(self.names):
            groups[self.labels[i]].append(name)
        return groups
    def get_labels(self):
        return self.labels
    def get_label(self, image_name):
        return self.labels[self.names.index(image_name)]

    def save(self, file_path):
        save_data = {}
        save_data['K'] = self.K
        save_data['loss'] = self.loss
        save_data['labels'] = self.labels
        save_data['centroids'] = self.centroids
        save_data['names'] = self.names

        print(f"Saving {file_path}...", end = '')
        with open(file_path, 'w') as json_file:
            json.dump(save_data, json_file)
        print("Done!")
    
    def load(self, load_path):
        print(f"Loading {load_path}...", end = '')
        with open(load_path, 'r') as file:
            save_data = json.load(file)
        print("Done!")
        self.K = save_data['K']
        self.loss = save_data['loss']
        self.labels = save_data['labels']
        self.centroids = save_data['centroids']
        self.names = save_data['names']



class ClosestImages:
    def __init__(self, centroids, preprocessed_datas):
        self.centroids = centroids
        self.preprocessed_datas = preprocessed_datas

    def get_closest_indicies(self, distance_fn):
        K = self.cluster_result['K']
        image_indicies = [-1 for i in range(K)]
        closest_dists = [-1 for i in range(K)]

        for itemid, item in tqdm(enumerate(self.preprocessed_datas)):
            for clusterid, centroid in enumerate(self.centroids):
                dist = distance_fn(item, centroid)
                if closest_dists[clusterid] == -1 or closest_dists[clusterid] > dist:
                    image_indicies[clusterid] = itemid
                    closest_dists[clusterid] = dist
        
        return image_indicies