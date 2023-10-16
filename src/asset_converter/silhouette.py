import sys
sys.path.append('./src')

import cv2
import json
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

from PoseFormatUtils import read_formatted_csv


if len(sys.argv) > 1:
    pidinet_dir = sys.argv[1]
    mask_dir = sys.argv[2]
    output_dir = sys.argv[3]

mask_threshold = 200
result_threshold = 64

def get_file_names_in_dir(dir_path):
    file_names = os.listdir(dir_path)
    return [file_name for file_name in file_names if os.path.isfile(os.path.join(dir_path, file_name))]

centroid_images = get_file_names_in_dir(pidinet_dir)

image_datas = []

for i, file_name in tqdm(enumerate(centroid_images)):
    file_name = file_name[:-4]
    silhouette = cv2.imread(f'{pidinet_dir}/{file_name}.png')
    mask = cv2.imread(f'{mask_dir}/mask_{file_name}.jpg', cv2.IMREAD_GRAYSCALE)

    original_size = (silhouette.shape[1], silhouette.shape[0])

    silhouette = cv2.resize(silhouette, (256,256))
    _, binary_mask = cv2.threshold(mask, mask_threshold, 255, cv2.THRESH_BINARY)

    masked_image = cv2.bitwise_and(silhouette, silhouette, mask = binary_mask)
    #masked_image[result_threshold <= 30] = 0

    smoothed_image = cv2.GaussianBlur(masked_image, (3, 3), 0)
    
    #원래 실루엣 크기로 복구
    smoothed_image = cv2.resize(smoothed_image, original_size)

    #바운딩박스 구하고, 실루엣을 바운딩박스에 맞게 자르기
    minx, miny, maxx, maxy = smoothed_image.shape[1], smoothed_image.shape[0], 0, 0
    for y in range(smoothed_image.shape[0]):
        for x in range(smoothed_image.shape[1]):
            pixel = smoothed_image[y,x]
            if pixel[0] == 0:
                continue
            minx = min(minx, x)
            miny = min(miny, y)
            maxx = max(maxx, x)
            maxy = max(maxy, y)

    sliced_silhouette = smoothed_image[miny:maxy+1, minx:maxx+1]

    totalwidth, totalheight = original_size[0], original_size[1]
    bbwidth, bbheight = maxx - minx, maxy - miny
    center_pos = (minx + bbwidth/2, maxy + bbheight/2)
    size = (bbwidth, bbheight)
    image_datas.append({'id':i, 'name':file_name,
                        'center':[center_pos[0] / totalwidth, center_pos[1] / totalheight],
                        'size':[size[0] / totalwidth, size[1] / totalheight]})
    #바운딩박스 center의 위치를 0~1 구해서 넣기, 크기도 0~1 구해서 넣기
    cv2.imwrite(f'{output_dir}/{i}_{file_name}.png', smoothed_image)

image_datas_df = pd.DataFrame(image_datas)
image_datas_df.to_csv(f'{output_dir}/image_datas.csv')