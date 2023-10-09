import cv2

from .pose_format import coco25
import configs

joints = coco25['parents']

#TEMP
keypoint_colors = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
           (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
           (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127),  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
           (77, 255, 255), (0, 255, 255), (77, 204, 255),  # head, neck, shoulder
           (0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0), (77, 255, 255)] # foot

def render_keypoints(cvimage, keypoints, withids = True, radius = 4 , font_size = 0.5, font_thickness = 2, line_thickness = 1, color = None, score_threshhold = 0.3):
        
    for i, point in enumerate(keypoints):
        position =  (int(point[0] * cvimage.shape[1]), int(point[1] * cvimage.shape[0]))
        score = point[2]
        if score < score_threshhold:
            continue

        #draw line
        parent_pos = int(keypoints[joints[i]][0] * cvimage.shape[1]), int(keypoints[joints[i]][1] * cvimage.shape[0])
        cur_color = color if color is not None else (255, 255, 255)
        cv2.line(cvimage, position, parent_pos, cur_color, line_thickness, cv2.LINE_AA)

        #draw point
        if color is None:
            cur_color = keypoint_colors[i]
        else:
            cur_color = color
        cv2.circle(cvimage, position, radius, cur_color, -1)
        if withids:
            cv2.putText(cvimage, str(i), position, cv2.FONT_HERSHEY_SIMPLEX, font_size,  keypoint_colors[i], font_thickness, cv2.LINE_AA)
    return cvimage