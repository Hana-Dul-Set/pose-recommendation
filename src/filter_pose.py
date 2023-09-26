from PoseFormatUtils import read_formatted_csv
from PoseFilterer.PoseFilterer import PoseFilterer

INPUT_PATH = "datas/formatted_pose_datas/alphapose_0924.csv"
OUTPUT_PATH = "datas/filtered_datas/vitpose_filtered_0926.csv"

formatted_df = read_formatted_csv(INPUT_PATH)

filterer = PoseFilterer(keypoint_existance_threshhold = 0.3)
filterer.must_have([0, 17, 18])
filterer.have_one_of([6, 8])
filterer.have_one_of([5, 7])
filterer.bbox_range(min_size = 0.05, max_size = 0.5)

filtered_df = filterer.filter(formatted_df, verbose=False)
filtered_df.to_csv(OUTPUT_PATH)