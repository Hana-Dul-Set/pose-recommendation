import json

import pandas as pd
from tqdm import tqdm

class VitPoseFileReader:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None

        self.read(csv_path)

    def read(self, csv_path):
        print(f"Reading {csv_path}...", end = '')
        self.df = pd.read_csv(csv_path, converters = {'keypoints':eval, 'size':eval, 'confidences':eval, 'bboxes':eval})
        print("Done!")

    def get_formatted_dataframe(self):
        result = []
        #,name,size,people_count,keypoints,bboxes,confidences
        for idx, row in tqdm(self.df.iterrows()):
            if row['people_count'] != 1:
                continue
            size = row['size']
            keypoints =  [[y / size[1], x / size[0], score] for x, y, score in row['keypoints']]
            bbox = row['bboxes'][0]

            parsed_data = {}
            parsed_data['name'] = row['name']
            parsed_data['size'] = size
            parsed_data['keypoints'] = keypoints
            parsed_data['confidence'] = row['confidences'][0]
            parsed_data['nose_pos'] = keypoints[0][:2]
            parsed_data['bbox'] = [bbox[0] / size[0], bbox[1] / size[1], bbox[2] / size[0], bbox[3] / size[1]]
            result.append(parsed_data)

        return pd.DataFrame(result)
    
    def save_formatted_dataframe(self, csv_path):
        df = self.get_formatted_dataframe()
        
        print(f"Saving to {csv_path}...", end = '')
        df.to_csv(csv_path)
        print(f"Done!")
        return df
