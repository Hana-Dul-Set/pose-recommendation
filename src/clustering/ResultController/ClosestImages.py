from tqdm import tqdm

class ClosestImages:
    def __init__(self, cluster_result, preprocessed_datas, formatted_df):
        self.cluster_result = cluster_result
        self.preprocessed_datas = preprocessed_datas
        self.formatted_df = formatted_df

    def get_closest_images(self, distance_fn):
        K = self.cluster_result['K']
        image_indicies = [-1 for i in range(K)]
        closest_dists = [-1 for i in range(K)]

        for itemid, item in tqdm(enumerate(self.preprocessed_datas)):
            for clusterid, centroid in enumerate(self.cluster_result['medoid_indicies']):
                dist = distance_fn(item, centroid)
                if closest_dists[clusterid] == -1 or closest_dists[clusterid] > dist:
                    image_indicies[clusterid] = itemid
                    closest_dists[clusterid] = dist
        
        return [self.formatted_df.loc[idx, 'name'] for idx in image_indicies]