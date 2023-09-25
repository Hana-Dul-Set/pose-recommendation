import pandas as pd
from tqdm import tqdm

class PoseFilterer:
    def __init__(self):
        self.filters = []

    def add_subfilter(self, subfilter):
        self.filters.append(subfilter)

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

