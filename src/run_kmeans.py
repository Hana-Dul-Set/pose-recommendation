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
from clustering.AnglePreprocessor import AnglePreprocessor
from clustering.Distances import distance, angle_distance
from PoseFormatUtils import read_formatted_csv
from clustering.KMeansResult import KMeansResult

save_path = f'datas/cluster_results/kmedoids_angle_200_test_1012.json'
K = 200

formatted_df = read_formatted_csv('datas/filtered_datas/vitpose_filtered_0927.csv')

preprocessor = AnglePreprocessor(formatted_df)
preprocessed_data = preprocessor.preprocess(existance_threshhold = 0.3)

starttime = datetime.now()
print("Start Time:", starttime.strftime("%H:%M:%S"))

clusterer = KMeansClustering(preprocessed_data, metric_fn = angle_distance, K = K)
clusterer.process()

duration = datetime.now() - starttime
print("End Time:", datetime.now().strftime("%H:%M:%S"))
print("Duraiton :", duration)

save_data = KMeansResult()
save_data.set_result(K, clusterer.get_total_error(), clusterer.get_labels(), clusterer.get_centers(), preprocessor.get_names())
save_data.save(save_path)