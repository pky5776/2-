from __future__ import print_function

import cv2
import numpy as np
from keras.models import Model
from keras.layers import Input, merge, concatenate, Dropout, Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K

from Data_new import load_train_data, load_test_data,load_test_data_1, load_test_data_2, load_test_data_3, load_test_data_4, \
                                                      load_test_data_5, load_test_data_6, load_test_data_7, load_test_data_8, \
                                                      load_test_data_9, load_test_data_10, load_test_data_11, load_test_data_12, load_test_data_13, \
                                                      load_test_data_14, load_test_data_15
from keras import callbacks
import os
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
  except RuntimeError as e:
    # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
    print(e)
remote=callbacks.RemoteMonitor(root='http://localhost:9000')

image_rows = 512
image_cols = 512

img_rows = 512
img_cols = 512

smooth = 1.
data_path = 'raw/'

def load_test_data():
    imgs_test = np.load('npy/imgs_test_new_data.npy')
    print(imgs_test.shape)
    return imgs_test

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def get_unet1():
    inputs = Input((img_rows, img_cols, 1))
    conv1 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Conv2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Conv2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Conv2D(512, 3, 3, activation='relu', border_mode='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=3)
    conv6 = Conv2D(256, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Conv2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=3)
    conv7 = Conv2D(128, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Conv2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=3)
    conv8 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=3)
    conv9 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Conv2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model

def get_unet2():
    inputs = Input((img_rows, img_cols, 1))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model

def preprocess_last(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols, 1), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        tmp = cv2.resize(imgs[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
        for m in range(img_rows):
            for n in range(img_cols):
                imgs_p[i][m][n][0] = tmp[m][n]
    return imgs_p

def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols, imgs.shape[1]), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i,0] = cv2.resize(imgs[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
    return imgs_p




def train_and_predict():

    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    imgs_test = load_test_data()
    imgs_test = preprocess_last(imgs_test)
    
    # imgs_test_1 = load_test_data_1()
    # imgs_test_1 = preprocess_last(imgs_test_1)
    
    # imgs_test_2 = load_test_data_2()
    # imgs_test_2 = preprocess_last(imgs_test_2)
    
    # imgs_test_3 = load_test_data_3()
    # imgs_test_3 = preprocess_last(imgs_test_3)
    
    # imgs_test_4 = load_test_data_4()
    # imgs_test_4 = preprocess_last(imgs_test_4)
    
    # imgs_test_5 = load_test_data_5()
    # imgs_test_5 = preprocess_last(imgs_test_5)
    # 20131021

    # imgs_test_6 = load_test_data_6()
    # imgs_test_6 = preprocess_last(imgs_test_6)
    
    # imgs_test_7 = load_test_data_7()
    # imgs_test_7 = preprocess_last(imgs_test_7)
    
    # imgs_test_8 = load_test_data_8()
    # imgs_test_8 = preprocess_last(imgs_test_8)
    # 20131028

    # imgs_test_9 = load_test_data_9()
    # imgs_test_9 = preprocess_last(imgs_test_9)
    
    # imgs_test_10 = load_test_data_10()
    # imgs_test_10 = preprocess_last(imgs_test_10)
    
    # imgs_test_11 = load_test_data_11()
    # imgs_test_11 = preprocess_last(imgs_test_11)
    # 20131114

    # imgs_test_12 = load_test_data_12()
    # imgs_test_12 = preprocess_last(imgs_test_12)
    
    # imgs_test_13 = load_test_data_13()
    # imgs_test_13 = preprocess_last(imgs_test_13)
    
    # imgs_test_14 = load_test_data_14()
    # imgs_test_14 = preprocess_last(imgs_test_14)
    # 20131121_Chip4

    # imgs_test_15 = load_test_data_15()
    # imgs_test_15 = preprocess_last(imgs_test_15)
    # 20131121_Chip6

    imgs_train, imgs_mask_train = load_train_data()



    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std
    model = get_unet2()
    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights('hdf5/New_Weight.hdf5')

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    imgs_mask_test = model.predict(imgs_test, verbose=1)
    np.save('npy/imgs_test_new_data.npy', imgs_mask_test)



if __name__ == '__main__':
    train_and_predict()
