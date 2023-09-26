import os

from PoseFormatUtils import read_formatted_csv

FILTERED_CSV_PATH = "datas/formatted_pose_datas/alphapose_0924.csv"
SOURCE_DIR = "datas/filtered_datas/vitpose_filtered_0926.csv"
TARGET_DIR = "datas/filtered_datas/images_vitpose_0926"

formatted_df = read_formatted_csv(FILTERED_CSV_PATH)
