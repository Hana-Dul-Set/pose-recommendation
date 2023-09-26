import pandas as pd

keys = ['name', 'size', 'keypoints', 'confidence', 'nose_pos', 'bbox']

def read_formatted_csv(csv_path):
    print(f"Reading dataframe: {csv_path}...", end = '')
    df = pd.read_csv(csv_path, converters={'keypoints' : eval, 'bbox' : eval, 'nose_pos' : eval, 'size' : eval})
    print("Done!")
    return df