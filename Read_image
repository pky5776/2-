# -*- coding: utf-8 -*- -
import cv2
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, Input
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Activation, Flatten
from matplotlib import pyplot as plt
import os
import numpy as np
import tensorflow as tf

data_path = 'u-net-demo/raw/'

img_rows = 300  # 입력영상 높이
img_cols = 300  # 입력영상 너비

def create_train_data():
    train_data_path = os.path.join(data_path, 'train')
    images = os.listdir(train_data_path)
    total = int(len(images) / 2)  # 2로 나누는 이유는 영상과 mask 2개가 1 set 이기 때문

    imgs = np.ndarray((total, img_rows, img_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, img_rows, img_cols), dtype=np.uint8)
    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:

        image_mask_name = image_name.split('.')[0] + '_mask.bmp'
        img = cv2.imread(os.path.join(train_data_path, image_name), cv2.IMREAD_GRAYSCALE)
        img_mask = cv2.imread(os.path.join(train_data_path, image_mask_name), cv2.IMREAD_GRAYSCALE)
        imgs[i] = img
        imgs_mask[i] = img_mask

    np.save('imgs_train.npy', imgs)
    np.save('imgs_mask_train.npy', imgs_mask)

def load_train_data():
    imgs_train = np.load('imgs_train.npy')
    imgs_mask_train = np.load('imgs_mask_train.npy')
    return imgs_train, imgs_mask_train
