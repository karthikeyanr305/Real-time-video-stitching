import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
plt.rcParams['figure.figsize'] = [15, 15]
from skimage import measure
import imutils
import time
from imutils.video import VideoStream
import cv2
from stitch import *

def main():
    cap1 = VideoStream(src=1).start()
    cap2 = VideoStream(src=0).start()
    time.sleep(50.0)
    img_array = []
    count = 0
    while True:
        frame11 = cap1.read()
        frame22 = cap2.read()

        time.sleep(10)

        frame1 = imutils.resize(frame11, width=400)
        frame2 = imutils.resize(frame22, width=400)

        frame1 = cv2.resize(frame1, (400,400), interpolation =cv2.INTER_AREA)
        frame2 = cv2.resize(frame2, (400,400), interpolation =cv2.INTER_AREA)

        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)


        left_gray, left_origin, left_rgb = read_image(frame1)
        right_gray, right_origin, right_rgb = read_image(frame2)

        kp_left, des_left = SIFT(left_gray)
        kp_right, des_right = SIFT(right_gray)

        kp_left_img = plot_sift(left_gray, left_rgb, kp_left)
        kp_right_img = plot_sift(right_gray, right_rgb, kp_right)

        """kp_left, des_left = ORB(left_gray)
        kp_right, des_right = ORB(right_gray)

        kp_left_img = plot_orb(left_gray, left_rgb, kp_left)
        kp_right_img = plot_orb(right_gray, right_rgb, kp_right)"""

        total_kp = np.concatenate((kp_left_img, kp_right_img), axis=1)

        matches = matcher(kp_left, des_left, left_rgb, kp_right, des_right, right_rgb, 0.5)

        total_img = np.concatenate((left_rgb, right_rgb), axis=1)

        inliers, H = ransac(matches, 0.5, 2000)

        final_image = stitch_img(left_rgb, right_rgb, H)

        #img_array.append(final_image)
        

        #final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        cv2.imshow("Result", final_image)
        cv2.imshow("Left Frame", frame1)
        cv2.imshow("Right Frame", frame2)
        #cv2.imshow("Matches",plot_matches(matches, total_img))
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    cap1.stop()
    cap2.stop()

main()

