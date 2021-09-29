from __future__ import print_function
import os
import numpy as np
import cv2
from Data import image_cols, image_rows
import matplotlib.pyplot as plt
import numpy as np

image_rows = 512
image_cols = 512


def prep(img):
    img = img.astype('float32')
    img = cv2.threshold(img, 28, 1., cv2.THRESH_BINARY)[1].astype(np.uint8)
    img = cv2.resize(img, (image_cols, image_rows))
    return img


def visualize():
    imgs_test = np.load('npy/imgs_test_new_data.npy')
    imgs_test_pred = np.load('npy/imgs_mask_test_1.npy')
    name = os.listdir("New_data/test_set")
    name_list = sorted(name)
    total_1 = int(len(name))
    total = imgs_test.shape[0]

    print(imgs_test_pred.shape)

    plt.figure(figsize=(10, 6))

    for N in range(total):

        gray_img = imgs_test[N, 0]

        w = 512
        h = 512

        t_img = np.zeros((h, w), np.float32)
        t_img2 = np.zeros((h, w), np.float32)

        for m in range(h):
            for n in range(w):
                t_img[m][n] = imgs_test_pred[N][m][n][0]

        for m in range(h):
            for n in range(w):
                t_img2[m][n] = imgs_test_pred[N][m][n][0]

        # print(t_img[256])

        bin_img = prep(t_img)

        rgb_img = np.ndarray((h, w, 3), dtype=np.uint8)

        for i in range(0, h, 1):
            for j in range(0, w, 1):
                rgb_img[i, j, 0] = gray_img[i, j]
                rgb_img[i, j, 1] = gray_img[i, j]
                rgb_img[i, j, 2] = gray_img[i, j]

        for i in range(0, h, 1):
            for j in range(0, w, 1):
                if bin_img[i, j] != 0:
                    rgb_img[i, j, 0] = 0
                    rgb_img[i, j, 2] = 0

        for i in range(1, h - 1, 1):
            for j in range(1, w - 1, 1):
                if bin_img[i, j] != 0 and gray_img[i, j] != 0:
                    if (bin_img[i - 1, j - 1] == 0
                            or bin_img[i - 1, j] == 0
                            or bin_img[i - 1, j + 1] == 0
                            or bin_img[i, j - 1] == 0
                            or bin_img[i, j + 1] == 0
                            or bin_img[i + 1, j - 1] == 0
                            or bin_img[i + 1, j] == 0
                            or bin_img[i + 1, j + 1] == 0):
                        rgb_img[i, j, 0] = 0
                        rgb_img[i, j, 1] = 255
                        rgb_img[i, j, 2] = 0
        print(N)

        cv2.imwrite("Result/Colorview/New/"+name_list[N], t_img * 255)


if __name__ == '__main__':
    visualize()