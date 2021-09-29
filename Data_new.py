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
# ______________________________________________________
def load_test_data_1():
    imgs_test = np.load('npy/20131017_Chip1.npy')
    return imgs_test

def load_test_data_2():
    imgs_test = np.load('npy/20131017_Chip2.npy')
    return imgs_test

def load_test_data_3():
    imgs_test = np.load('npy/20131017_Chip3.npy')
    return imgs_test

def load_test_data_4():
    imgs_test = np.load('npy/20131017_Chip5.npy')
    return imgs_test
# ______________________________________________________
def load_test_data_5():
    imgs_test = np.load('npy/20131021_Chip1.npy')
    return imgs_test

def load_test_data_6():
    imgs_test = np.load('npy/20131021_Chip2.npy')
    return imgs_test

def load_test_data_7():
    imgs_test = np.load('npy/20131021_Chip3.npy')
    return imgs_test
# ______________________________________________________
def load_test_data_8():
    imgs_test = np.load('npy/20131028_Chip1.npy')
    return imgs_test

def load_test_data_9():
    imgs_test = np.load('npy/20131028_Chip2.npy')
    return imgs_test

def load_test_data_10():
    imgs_test = np.load('npy/20131028_Chip3.npy')
    return imgs_test
# ______________________________________________________
def load_test_data_11():
    imgs_test = np.load('npy/20131114_Chip1.npy')
    return imgs_test

def load_test_data_12():
    imgs_test = np.load('npy/20131114_Chip2.npy')
    return imgs_test

def load_test_data_13():
    imgs_test = np.load('npy/20131114_Chip3.npy')
    return imgs_test
# ______________________________________________________
def load_test_data_14():
    imgs_test = np.load('npy/20131121_Chip4.npy')
    return imgs_test

def load_test_data_15():
    imgs_test = np.load('npy/20131121_Chip6.npy')
    return imgs_test 
 

if __name__ == '__main__':
    create_train_data()
    load_test_data()
    # load_test_data_1()
    # load_test_data_2()
    # load_test_data_3()
    # load_test_data_4()
    # load_test_data_5()
    # load_test_data_6()
    # load_test_data_7()
    # load_test_data_8()
    # load_test_data_9()
    # load_test_data_10()
    # load_test_data_11()
    # load_test_data_12()
    # load_test_data_13()
    # load_test_data_14()
    # load_test_data_15()