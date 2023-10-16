import sys
sys.path.append('./src')

import json
import os
from datetime import datetime

import kmedoids

from clustering.Distances import distance
from PoseFormatUtils import read_formatted_csv
from clustering.Preprocessor import Preprocessor
from clustering.KMedoidsResult import KMedoidsResult

K = 200
RECALCULATE = False
method = 'pam'
save_path = f'datas/cluster_results/kmedoids_{method}_{K}_test_1010.json'



intermediate_dir = 'datas/intermediate_datas'
precalculated_distances_path = os.path.join(intermediate_dir, 'precalc_distances_1010.json')


formatted_df = read_formatted_csv('datas/filtered_datas/vitpose_filtered_0927.csv')

preprocessor = Preprocessor(formatted_df)
preprocessor.preprocess(existance_threshhold = 0.5)

if RECALCULATE:
   preprocessor.precalculate_distances(distance, precalculated_distances_path)
else:
   preprocessor.load_precalculated_distances(precalculated_distances_path)

distance_matrix = preprocessor.get_distance_matrix()
starttime = datetime.now()
print("Start Time:", starttime.strftime("%H:%M:%S"))

km = kmedoids.KMedoids(K, method = method)
result = km.fit(distance_matrix)

duration = datetime.now() - starttime
print("End Time:", datetime.now().strftime("%H:%M:%S"))
print("Duraiton :", duration)

save_data = KMedoidsResult()
save_data.set_result(K, result.inertia_, result.labels_.tolist(), result.medoid_indices_.tolist(), preprocessor.get_names())
save_data.save(save_path)
