import sys
sys.path.append('./src')

import json
import math
import time
import os
from datetime import datetime
from tqdm import tqdm

from clustering.KMedoids import KMedoidsClustering
from clustering.KMedians import KMediansClustering
from clustering.KMeans import KMeansClustering
from clustering.Preprocessor import Preprocessor
from clustering.Distances import distance
from PoseFormatUtils import read_formatted_csv

intermediate_dir = 'datas/intermediate_datas'
precalculated_distances_path = os.path.join(intermediate_dir, 'precalc_distances_1004.json')

RECALCULATE = False
K = 60

formatted_df = read_formatted_csv('datas/filtered_datas/vitpose_filtered_0927.csv')

preprocessor = Preprocessor(formatted_df)
preprocessor.preprocess(existance_threshhold = 0.3)

if RECALCULATE:
   preprocessor.precalculate_distances(distance, precalculated_distances_path)
else:
   preprocessor.load_precalculated_distances(precalculated_distances_path)

def indexed_distance(A, B):
   return preprocessor.get_precalculated_distance(A, B)

indicies = [i for i in range(len(formatted_df))]

starttime = datetime.now()
print("Start Time:", starttime.strftime("%H:%M:%S"))

clusterer = KMedoidsClustering(indicies, metric_fn = indexed_distance, K = K)
clusterer.process()

duration = datetime.now() - starttime
print("End Time:", datetime.now().strftime("%H:%M:%S"))
print("Duraiton :", duration)

#save result
save_data ={}
save_data['clusters'] = clusterer.get_clusters()
save_data['medoid_indicies'] = clusterer.get_medoid_indicies()
save_data['names'] = preprocessor.get_names()
#save_data['total_wec'] = clusterer.get_total_error()
save_data['K'] = K


result_path = f'datas/cluster_results/kmedoids_{K}_test_1004.json'

print(f"Saving {result_path}...", end = '')
with open(result_path, 'w') as json_file:
   json.dump(save_data, json_file)
print("Done!")