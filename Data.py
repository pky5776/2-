from __future__ import print_function

import os
import numpy as np

import cv2

data_path = 'New_data'

image_rows = 512
image_cols = 512


def create_train_data():
    train_data_path = os.path.join(data_path, 'train_set')
    images = os.listdir(train_data_path)
    total = int(len(images) / 2)

    imgs = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)

    i = 0
    print('-' * 30)
    print('Creating training images...')
    print('-' * 30)
    for image_name in images:
        if 'mask' in image_name:
            continue
        image_mask_name = image_name.split('.')[0] + '_mask.png'
        img = cv2.imread(os.path.join(train_data_path, image_name), cv2.IMREAD_GRAYSCALE)
        img_mask = cv2.imread(os.path.join(train_data_path, image_mask_name), cv2.IMREAD_GRAYSCALE)

        img = np.array([img])
        img_mask = np.array([img_mask])

        imgs[i] = img
        imgs_mask[i] = img_mask

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save('npy/imgs_train_new_data.npy', imgs)
    np.save('npy/imgs_mask_train_new_data.npy', imgs_mask)
    print('Saving to .npy files done.')


def load_train_data():
    imgs_train = np.load('npy/imgs_train_new_data.npy')
    imgs_mask_train = np.load('npy/imgs_mask_train_new_data.npy')
    return imgs_train, imgs_mask_train


def create_test_data():
    test_data_path_1 = os.path.join(data_path, 'test_set')
    images_1 = os.listdir(test_data_path_1)
    images_list = sorted(images_1)
    total_1 = len(images_1)

    imgs_1 = np.ndarray((total_1, 1, image_rows, image_cols), dtype=np.uint8)
    imgs_id_1 = np.ndarray((total_1,), dtype=np.int32)

    print('-' * 30)
    print('Creating test images...')
    print('-' * 30)
    i = 0
    for image_name in images_list:
        img = cv2.imread(os.path.join(test_data_path_1, image_name), cv2.IMREAD_GRAYSCALE)
        imgs_1[i] = img
        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total_1))
        i += 1
    print('Loading done.')
    np.save('npy/imgs_test_new_data.npy', imgs_1)

    print('Saving to .npy files done.')

def load_test_data():
    imgs_test = np.load('npy/imgs_test_new_data.npy')
    return imgs_test
 

if __name__ == '__main__':
    create_train_data()
    create_test_data()
