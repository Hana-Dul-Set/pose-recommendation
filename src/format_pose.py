#from PoseFileReader.AlphaPoseFileReader import AlphaPoseFileReader
from PoseFileReader.ViTPoseFileReader import VitPoseFileReader

INPUT_PATH = "datas/pose_estimation_results/vitpose-keypoints0925.csv"
OUTPUT_PATH = "datas/formatted_pose_datas/vitpose_0927.csv"

data_reader = VitPoseFileReader(INPUT_PATH)
df = data_reader.save_formatted_dataframe(OUTPUT_PATH)
