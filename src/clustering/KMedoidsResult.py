import json

class KMedoidsResult:
    def __init__(self, load_path = ""):
        if load_path != "":
            self.load(load_path)
    
    def set_result(self, K, loss, labels, medoid_indicies, names):
        self.K = K
        self.loss = loss
        self.labels = labels
        self.medoid_indicies = medoid_indicies
        self.names = names

    def get_loss(self):
        return self.loss
    
    def get_names(self):
        return self.names
    
    def get_K(self):
        return self.K
    
    def get_medoid_indicies(self):
        return self.medoid_indicies
    
    def get_medoid_images(self):
        medoids = []
        for i in range(self.K):
            medoids.append(self.names[self.medoid_indicies[i]])
        return medoids
        
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
        save_data['medoid_indicies'] = self.medoid_indicies
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
        self.medoid_indicies = save_data['medoid_indicies']
        self.names = save_data['names']