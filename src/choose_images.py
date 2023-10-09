import os
import shutil
from tqdm import tqdm

from PoseFormatUtils import read_formatted_csv

FILTERED_CSV_PATH = "datas/filtered_datas/vitpose_filtered_0927.csv"
SOURCE_DIR = "datas/all_images_resized"
TARGET_DIR = "datas/filtered_datas/images_vitpose_filtered_0927"    

formatted_df = read_formatted_csv(FILTERED_CSV_PATH)

os.makedirs(TARGET_DIR, exist_ok=True)

for name in tqdm(formatted_df['name']):
    source_file_path = os.path.join(SOURCE_DIR, name)
    destination_file_path = os.path.join(TARGET_DIR, name)

    # Check if the source file exists
    if os.path.exists(source_file_path):
        shutil.copy(source_file_path, destination_file_path)
    else:
        print(f"{name} doesn't exist")