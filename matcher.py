import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import draw_polygon_on_image, save_image


def match_features(query_gray, target_gray):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(query_gray, None)
    kp2, des2 = sift.detectAndCompute(target_gray, None)

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    return kp1, des1, kp2, des2, good_matches

def match_and_display_image(query_path, target_path, output_path='results/result_image.jpg'):
    query_color = cv2.imread(query_path)
    target_color = cv2.imread(target_path)

    query_gray = cv2.cvtColor(query_color, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_color, cv2.COLOR_BGR2GRAY)

    kp1, _, kp2, _, good_matches = match_features(query_gray, target_gray)

    if len(good_matches) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M is not None:
            result_img = draw_polygon_on_image(query_color, target_color, kp1, kp2, good_matches, M, mask)
            save_image(result_img, output_path)
            plt.figure(figsize=(15, 10))
            plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
            plt.title("Matched Keypoints and Bounding Box")
            plt.axis("off")
            plt.show()
        else:
            print("Homography could not be computed.")
    else:
        print("Not enough good matches to draw bounding box.")

def match_and_save_video(query_path, video_path, output_path='results/result_video.mp4'):
    # Load query image and extract features
    query_color = cv2.imread(query_path)
    query_gray = cv2.cvtColor(query_color, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(query_gray, None)

    # Setup matcher
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video.")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 25  # default fallback

    # Ensure results folder exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Define the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp2, des2 = sift.detectAndCompute(frame_gray, None)

        if des2 is None:
            out.write(frame)
            continue

        matches = flann.knnMatch(des1, des2, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good_matches) > 10:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is not None:
                h, w = query_color.shape[:2]
                pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)
                boxed = cv2.polylines(frame.copy(), [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
                out.write(boxed)
            else:
                out.write(frame)
        else:
            out.write(frame)

    cap.release()
    out.release()
    print(f"Output saved at: {output_path}")