import json

import pandas as pd
from tqdm import tqdm

class AlphaPoseFileReader:
    def __init__(self, json_path):
        self.file_path = json_path
        self.json_data = None

        self.read(json_path)

    def read(self, json_path):
        print(f"Reading {json_path}...", end = '')
        with open(json_path, 'r') as json_file:
            self.json_data = json.load(json_file)
        print("Done!")

    def get_formatted_dataframe(self):
        result = []
        for data in tqdm(self.json_data):
            if data['people_count'] != 1:
                continue
            keypoints = data['keypoints'][0]

            parsed_data = {}
            parsed_data['name'] = data['name']
            parsed_data['size'] = data['size']
            parsed_data['keypoints'] = keypoints
            parsed_data['confidence'] = data['person_scores'][0]
            parsed_data['nose_pos'] = keypoints[0][:2]
            parsed_data['bbox'] = self.find_bounding_box(keypoints)
            result.append(parsed_data)

        return pd.DataFrame(result)
    
    def save_formatted_dataframe(self, csv_path):
        df = self.get_formatted_dataframe()
        
        print(f"Saving to {csv_path}...", end = '')
        df.to_csv(csv_path)
        print(f"Done!")
        return df

    def find_bounding_box(self, keypoints):
        # Initialize min and max values with the first coordinate
        min_x, min_y, s = keypoints[0]
        max_x, max_y, s = keypoints[0]

        # Iterate through the coordinates to update min and max values
        for x, y, s in keypoints:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)

        # Return the bounding box as a tuple (min_x, min_y, max_x, max_y)
        return [min_x, min_y, max_x, max_y]