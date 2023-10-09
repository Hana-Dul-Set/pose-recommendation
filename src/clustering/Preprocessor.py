from tqdm import tqdm
import pandas as pd
import json

from vis.pose_format import coco25

parents = coco25['parents']

class Preprocessor:
    def __init__(self, formatted_data_df):
        self.formatted_data_df = formatted_data_df
    
    def preprocess(self, existance_threshhold = 0.3):
        self.preprocessed_datas = []
        for idx, row in tqdm(self.formatted_data_df.iterrows()):
            nose_pos = row['nose_pos']
            keypoints = row['keypoints']

            #make relative
            relative_pose = [[0, 0, 0] for i in range(len(keypoints))]
            for i in range(len(keypoints)):
                score = keypoints[i][2]
                if score < existance_threshhold:
                    relative_pose[i][2] = 1
                relative_pose[i][0] = keypoints[i][0] - keypoints[parents[i]][0]
                relative_pose[i][1] = keypoints[i][1] - keypoints[parents[i]][1]
            
            #flattened
            result = [nose_pos] + relative_pose
            result = [item for sublist in result for item in sublist]

            self.preprocessed_datas.append(result)
        return self.preprocessed_datas

    def precalculate_distances(self, distance_fn, output_file_path):
        print("Start precalculating distances...")
        self.distance_datas = {}
        for idx1 in tqdm(range(len(self.preprocessed_datas))):
            self.distance_datas[idx1] = {}
            for idx2 in tqdm(range(idx1+1, len(self.preprocessed_datas)), leave = False):
                item1 = self.preprocessed_datas[idx1]
                item2 = self.preprocessed_datas[idx2]
                self.distance_datas[idx1][idx2] = distance_fn(item1, item2)

        self.distance_datas['names'] = [row['name'] for idx, row in self.formatted_data_df.iterrows()]
        print(f"Precalcuation Done! Start saving to {output_file_path}...", end = '')
        with open(output_file_path, 'w') as file:
            json.dump(self.distance_datas, file)
        print("Done!")
        

    def load_precalculated_distances(self, file_path):
        print(f"Loading {file_path}...", end = '')
        with open(file_path, 'r') as file:
            self.distance_datas = json.load(file)
        print("Done!")
    
    def get_precalculated_distance(self, index1, index2):
        if index1 == index2:
            return 0
        if index1 > index2:
            index1, index2 = index2, index1
        return self.distance_datas[str(index1)][str(index2)]

    def get_names(self):
        return self.distance_datas['names']