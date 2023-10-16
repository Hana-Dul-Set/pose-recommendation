import sys
sys.path.append('./src')

import json
import os
from datetime import datetime

import kmedoids

from clustering.Distances import distance
from PoseFormatUtils import read_formatted_csv
from clustering.AnglePreprocessor import AnglePreprocessor
from clustering.KMedoidsResult import KMedoidsResult

K = 5
RECALCULATE = True
method = 'fasterpam'
save_path = f'datas/cluster_results/kmedoids_angle_{method}_{K}_test_1013.json'

intermediate_dir = 'datas/intermediate_datas'
precalculated_distances_path = os.path.join(intermediate_dir, 'precalc_angle_distances_1013.json')


formatted_df = read_formatted_csv('datas/filtered_datas/vitpose_filtered_0927.csv')[:100]

preprocessor = AnglePreprocessor(formatted_df)
preprocessor.preprocess(existance_threshhold = 0.5)

if RECALCULATE:
   preprocessor.precalculate_distances(precalculated_distances_path)
else:
   preprocessor.load_precalculated_distances(precalculated_distances_path)

distance_matrix = preprocessor.get_distance_matrix()

print(distance_matrix)
print(len(distance_matrix))
print(len(distance_matrix[0]))
starttime = datetime.now()
print("Start Time:", starttime.strftime("%H:%M:%S"))

result = kmedoids.pam(distance_matrix, medoids=K, init='build')

duration = datetime.now() - starttime
print("End Time:", datetime.now().strftime("%H:%M:%S"))
print("Duraiton :", duration)

save_data = KMedoidsResult()
save_data.set_result(K, result.loss, result.labels.tolist(), result.medoids.tolist(), preprocessor.get_names())
save_data.save(save_path)
