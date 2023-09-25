from PoseFileReader.AlphaPoseFileReader import AlphaPoseFileReader

INPUT_PATH = "datas/pose_estimation_results/alphapose/all_photo_data_0622.json"
OUTPUT_PATH = "datas/formatted_pose_datas/alphapose_0924.csv"

data_reader = AlphaPoseFileReader(INPUT_PATH)
df = data_reader.save_formatted_dataframe(OUTPUT_PATH)
