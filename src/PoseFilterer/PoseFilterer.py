import pandas as pd
from tqdm import tqdm

from .Subfilters.Subfilters import MustHaveKeypoints, HaveOneOfKeypoints, BBoxSizeRange, MinConfidence

class PoseFilterer:
    def __init__(self, keypoint_existance_threshhold = 0.3):
        self.filters = []
        self.keypoint_existance_threshhold = keypoint_existance_threshhold

    def add_subfilter(self, subfilter):
        self.filters.append(subfilter)

    def min_confidence(self, confidence_threshhold):
        self.add_subfilter(MinConfidence(confidence_threshhold))

    def must_have(self, keypoint_list):
        self.add_subfilter(MustHaveKeypoints(keypoint_list, self.keypoint_existance_threshhold))

    def have_one_of(self, keypoint_list):
        self.add_subfilter(HaveOneOfKeypoints(keypoint_list, self.keypoint_existance_threshhold))

    def bbox_range(self, min_size, max_size):
        self.add_subfilter(BBoxSizeRange(min_size, max_size))


    def filter(self, formatted_dataframe, verbose = False):
        filtered_rows = []
        print("Filtering Start!")
        for index, row in tqdm(formatted_dataframe.iterrows()):
            include = True
            for subfilter in self.filters:
                if subfilter.should_remove(row):
                    include = False
                    if verbose:
                        print(row['name'],":",subfilter)
                    break
            
            if not include:
                continue
            filtered_rows.append(row)
        print("Filtering Done!")
        print(f"{len(filtered_rows)} rows left after filtering {len(formatted_dataframe)} rows.")
        return pd.DataFrame(filtered_rows)

