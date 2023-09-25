
#keypoint_list에 적힌 모든 index들의 score가 threshhold보다 높아야함
class MustHaveKeypoints:
    def __init__(self, keypoint_list, threshhold):
        self.keypoint_list = keypoint_list
        self.threshhold = threshhold

    def should_remove(self, row):
        for point in self.keypoint_list:
            if row['keypoints'][point][2] < self.threshhold:
                return True
        return False
    
    def __str__(self):
        return f"MustHaveKeypoints[{self.keypoint_list}]"

#keypoint_list index 중 하나라도 score가 threshhold보다 높아야함
class HaveOneOfKeypoints:
    def __init__(self, keypoint_list, threshhold):
        self.keypoint_list = keypoint_list
        self.threshhold = threshhold

    def should_remove(self, row):
        for point in self.keypoint_list:
            if row['keypoints'][point][2] > self.threshhold:
                return False
        return True
    
    def __str__(self):
        return f"HaveOneOfKeypoints[{self.keypoint_list}]"
        
#bbox가 min이상 max이하가 되어야함
class BBoxSizeRange:
    def __init__(self, min_size = 0, max_size = 1):
        self.min_size = min_size
        self.max_size = max_size
    
    def should_remove(self, row):
        bbox = row['bbox']
        size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        return not (self.min_size <= size <= self.max_size)
    
    def __str__(self):
        return f"BBoxSizeRange[{self.min_size}, {self.max_size}]"