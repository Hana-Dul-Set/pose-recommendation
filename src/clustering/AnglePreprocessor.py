import sys
sys.path.append('src/')
from tqdm import tqdm
import pandas as pd
import json
import math
from vis.pose_format import coco25
from clustering.Distances import angle_distance

parents = coco25['parents']

angle_triplets = [(3, 1, 0), (4, 2, 0), (6, 5, 0), (7, 5, 0), (8, 6, 5), (9, 7, 5), (10, 8, 6), (11, 9, 7), (12, 14, 5),
                   (13, 14, 5), (14, 5, 0), (15, 12, 14), (16, 13, 14), (17, 15, 12), (18, 16, 13), (19, 21, 17), (20, 19, 21), 
                   (21, 17, 15), (22, 24, 18), (23, 22, 24), (24, 18, 16)]
'''for idx in parents.keys():
   p = parents[idx]
   pp = parents[p]
   if idx == p or p == pp:
      continue
   else:
      angle_triplets.append((idx, p, pp))

print(angle_triplets)'''

class AnglePreprocessor:
    def __init__(self, formatted_data_df):
        self.formatted_data_df = formatted_data_df
    
    def preprocess(self, existance_threshhold = 0.3):
        self.preprocessed_datas = []
        for idx, row in tqdm(self.formatted_data_df.iterrows()):
            nose_pos = row['nose_pos']
            keypoints = row['keypoints']

            #angles
            angles = []
            for i in range(len(keypoints)):
                if keypoints[i][2] < existance_threshhold:
                    keypoints[i] = [keypoints[parents[i]][0], keypoints[parents[i]][1], 1]
                else:
                    keypoints[i][2] = 0

            for a, b, c in angle_triplets:
                angle = calculate_angle(keypoints[a], keypoints[b], keypoints[c]) / 180
                angles.append([angle, keypoints[b][2]])
            
            #flattened
            result = [nose_pos] + angles
            result = [item for sublist in result for item in sublist]

            self.preprocessed_datas.append(result)
        return self.preprocessed_datas
    
    def precalculate_distances(self, output_file_path):
        print("Start precalculating distances...")
        self.distance_datas = {}
        for idx1 in tqdm(range(len(self.preprocessed_datas))):
            self.distance_datas[str(idx1)] = {}
            for idx2 in tqdm(range(idx1+1, len(self.preprocessed_datas)), leave = False):
                item1 = self.preprocessed_datas[idx1]
                item2 = self.preprocessed_datas[idx2]
                self.distance_datas[str(idx1)][str(idx2)] = angle_distance(item1, item2)

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

    def get_distance_matrix(self):
        size = len(self.distance_datas['names'])
        distance_matrix = [[0 for j in range(size)] for i in range(size)]
        for idx1 in tqdm(range(size)):
            for idx2 in range(size):
                distance_matrix[idx1][idx2] = self.get_precalculated_distance(idx1, idx2)
        return distance_matrix
    
    def get_precalculated_distance(self, index1, index2):
        if index1 == index2:
            return 0
        if index1 > index2:
            index1, index2 = index2, index1
        return self.distance_datas[str(index1)][str(index2)]

    def get_names(self):
        return self.distance_datas['names']

def calculate_angle(A, B, C):
    vector_AB = [ A[0] - B[0], A[1] - B[1]]
    vector_BC = [C[0] - B[0], C[1] - B[1]]
    dot_product = vector_AB[0] * vector_BC[0] + vector_AB[1] * vector_BC[1]
    magnitude_AB = math.sqrt(vector_AB[0] ** 2 + vector_AB[1] ** 2)
    magnitude_BC = math.sqrt(vector_BC[0] ** 2 + vector_BC[1] ** 2)
    if (magnitude_AB * magnitude_BC) == 0:
        return 0
    # Calculate the cosine of the angle using the dot product and magnitudes
    cosine_theta = dot_product / (magnitude_AB * magnitude_BC)
    # Calculate the angle in radians using the arccosine function
    angle_radians = math.acos(cosine_theta)
    angle_degrees = math.degrees(angle_radians)
    
    return angle_degrees

