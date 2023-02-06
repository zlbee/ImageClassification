import cv2
import pickle


# https://stackoverflow.com/questions/67762285/drawing-sift-keypoints
def draw_cross_keypoints(img, keypoints, color):
    """ Draw keypoints as crosses, and return the new image with the crosses. """
    img_kp = img.copy()  # Create a copy of img

    # Iterate over all keypoints and draw a cross on evey point.
    for kp in keypoints:
        x, y = kp.pt  # Each keypoint as an x, y tuple  https://stackoverflow.com/questions/35884409/how-to-extract-x-y-coordinates-from-opencv-cv2-keypoint-object

        x = int(round(x))  # Round an cast to int
        y = int(round(y))

        # Draw a cross with (x, y) center
        cv2.drawMarker(img_kp, (x, y), color, markerType=cv2.MARKER_CROSS, markerSize=1, thickness=1, line_type=cv2.LINE_8)

    return img_kp  # Return the image with the drawn crosses.

# step 1: feature extraction
# # SIFT: https://medium.com/@aybukeyalcinerr/bag-of-visual-words-bovw-db9500331b2f
import matplotlib.pyplot as plt
import numpy as np


def find_sifts_images(images):
    print('SIFT descriptors finding started.')
    all_descriptors = []
    all_descriptors_images = []
    sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.05,
                                       sigma=1.6,
                                       edgeThreshold=10,
                                       nOctaveLayers=4)
    for image in images:
        kp, des = sift.detectAndCompute(image, None)
        if des is not None:
            all_descriptors.extend(des)
            all_descriptors_images.append(des)
            print(len(kp))
            # siftimg = cv2.drawKeypoints(image, kp, image)
            # plt.imshow(siftimg)
            # plt.show()
            # orbimg = draw_cross_keypoints(image, kp, color=(0, 0, 255))
            # plt.imshow(orbimg, cmap='gray')
            # plt.show()
        else:
            all_descriptors_images.append(None)
    print('SIFT descriptors finding ended.')
    return all_descriptors, all_descriptors_images


# # SURF
def find_surfs_images(images, run_mode, train_mode='train'):
    print('SURF descriptors finding started.')
    all_descriptors = []
    all_descriptors_images = []
    if run_mode == 'train':
        surf = cv2.xfeatures2d.SURF_create()
        surf.setExtended(True)
        for image in images:
            kp, des = surf.detectAndCompute(image, None)
            if des is not None:
                all_descriptors.extend(des)
                all_descriptors_images.append(des)
                print(len(kp))
                # surfimg = cv2.drawKeypoints(image, kp, None, (255, 0, 0), 4)
                # surfimg = cv2.drawKeypoints(image, kp, image)
                # plt.imshow(surfimg)
                # plt.show()
            else:
                all_descriptors_images.append(None)
    print('SIFT descriptors finding ended.')
    return all_descriptors, all_descriptors_images


def find_orbs_images(images, run_mode, train_mode='train'):
    print('ORB descriptors finding started.')
    all_descriptors = []
    all_descriptors_images = []
    if run_mode == 'train':
        orb = cv2.ORB_create(edgeThreshold=5, patchSize=5, nlevels=8, fastThreshold=20, scaleFactor=1.1, WTA_K=2,scoreType=cv2.ORB_HARRIS_SCORE, firstLevel=0)
        for image in images:
            kp = orb.detect(image, None)
            kp, des = orb.compute(image, kp)
            if des is not None:
                all_descriptors.extend(des)
                all_descriptors_images.append(des)
                print(len(kp))
                # surfimg = cv2.drawKeypoints(image, kp, image)
                # plt.imshow(surfimg)
                # plt.show()
                # orbimg = draw_cross_keypoints(image, kp, color=(0, 0, 255))
                # plt.imshow(orbimg, cmap='gray')
                # plt.show()
            else:
                all_descriptors_images.append(None)
    print('ORB descriptors finding ended.')
    return all_descriptors, all_descriptors_images