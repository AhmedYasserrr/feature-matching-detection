import cv2
import numpy as np
import os

def draw_polygon_on_image(query_img, target_img, kp1, kp2, matches, homography, mask):
    h, w = query_img.shape[:2]
    pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, homography)

    boxed_img = target_img.copy()
    boxed_img = cv2.polylines(boxed_img, [np.int32(dst)], True, (0, 255, 0), 4, cv2.LINE_AA)

    result_img = cv2.drawMatches(query_img, kp1, boxed_img, kp2, matches, None,
                                 matchColor=(255, 0, 0), singlePointColor=None,
                                 matchesMask=mask.ravel().tolist(), flags=2)
    return result_img

def save_image(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)