import sys
sys.path.append('./src')

import cv2
import json

from PoseFormatUtils import read_formatted_csv

with open("datas/output_assets/centroid_images.json", "r") as file:
    centroid_images = json.load(file)


for file_name in centroid_images:
    file_name = file_name[:-4]
    silhouette = cv2.imread(f'datas/silhouettes/{file_name}.png')
    mask = cv2.imread(f'datas/silhouette_masks/mask_{file_name}.jpg', cv2.IMREAD_GRAYSCALE)

    silhouette = cv2.resize(silhouette, (256,256))

    _, binary_mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    masked_image = cv2.bitwise_and(silhouette, silhouette, mask = binary_mask)



    cv2.imshow('masked', masked_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()