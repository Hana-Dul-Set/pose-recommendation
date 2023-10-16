import sys
sys.path.append('C:/YJ/soma/pose-recommendation/src/')
import json

#from clustering.ResultController.ClosestImages import ClosestImages
from clustering.Preprocessor import Preprocessor
from clustering.KMeansResult import KMeansResult
from clustering.KMedoidsResult import KMedoidsResult
from clustering.Distances import distance
from PoseFormatUtils import read_formatted_csv

formatted_df = read_formatted_csv('datas/filtered_datas/vitpose_filtered_0927.csv')

if len(sys.argv) > 1:
    result_path = sys.argv[1]
    target_dir = sys.argv[2]
    if target_dir[-1] != '/':
        target_dir += '/'
else:
    result_path = "datas/cluster_results/kmedoids_pam_200_test_1010.json"
    dir_id = result_path.split('/')[-1].split('.')[0]
    target_dir = 'C:/YJ/soma/pose-recommendation/datas/intermediate_datas/silhouette_inputs/'+dir_id+'/'

dir_id = result_path.split('/')[-1].split('.')[0]

if 'medoids' in dir_id: 
    result = KMedoidsResult(result_path)
    medoids = result.get_medoid_images()
elif 'means' in dir_id:
    result = KMeansResult(result_path)
    medoids = result.get_centroid_images()
    
import os
import shutil

images_dir = 'datas/all_images_resized/'

print(f"Copying {len(medoids)} images to {target_dir}")

for file_name in medoids:
    shutil.copy(images_dir + file_name, target_dir + file_name)