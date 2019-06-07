"""

this is a wrapper handle CASIA_HWDB dataset
since original data is complicated
we using this class to get .png and label from raw
.gnt data

"""
from alfred.dl.tf.common import mute_tf
mute_tf()
import struct
import numpy as np
import cv2
import tensorflow as tf

import os


this_dir = os.path.dirname(os.path.abspath(__file__))


def parse_example(record):
    features = tf.io.parse_single_example(record,
                                          features={
                                              'label':
                                                  tf.io.FixedLenFeature([], tf.int64),
                                              'image':
                                                  tf.io.FixedLenFeature([], tf.string),
                                          })
    img = tf.io.decode_raw(features['image'], out_type=tf.uint8)
    img = tf.cast(tf.reshape(img, (64, 64)), dtype=tf.float32)
    label = tf.cast(features['label'], tf.int64)
    return {'image': img, 'label': label}


def parse_example_v2(record):
    """
    latest version format
    :param record:
    :return:
    """
    features = tf.io.parse_single_example(record,
                                          features={
                                              'width':
                                                  tf.io.FixedLenFeature([], tf.int64),
                                              'height':
                                                  tf.io.FixedLenFeature([], tf.int64),
                                              'label':
                                                  tf.io.FixedLenFeature([], tf.int64),
                                              'image':
                                                  tf.io.FixedLenFeature([], tf.string),
                                          })
    img = tf.io.decode_raw(features['image'], out_type=tf.uint8)
    # we can not reshape since it stores with original size
    w = features['width']
    h = features['height']
    img = tf.cast(tf.reshape(img, (w, h)), dtype=tf.float32)
    label = tf.cast(features['label'], tf.int64)
    return {'image': img, 'label': label}


def load_ds():
    input_files = ['dataset/HWDB1.1trn_gnt.tfrecord']
    ds = tf.data.TFRecordDataset(input_files)
    ds = ds.map(parse_example)
    return ds


def load_val_ds():
    input_files = ['dataset/HWDB1.1tst_gnt.tfrecord']
    ds = tf.data.TFRecordDataset(input_files)
    ds = ds.map(parse_example_v2)
    return ds


def load_characters():
    a = open(os.path.join(this_dir, 'characters.txt'), 'r').readlines()
    return [i.strip() for i in a]


if __name__ == "__main__":
    ds = load_ds()
    val_ds = load_val_ds()
    val_ds = val_ds.shuffle(100)
    charactors = load_characters()

    is_show_combine = False
    if is_show_combine:
        combined = np.zeros([32*10, 32*20], dtype=np.uint8)
        i = 0
        res = ''
        for data in val_ds.take(200):
            # start training on model...
            img, label = data['image'], data['label']
            img = img.numpy()
            img = np.array(img, dtype=np.uint8)
            img = cv2.resize(img, (32, 32))
            label = label.numpy()
            label = charactors[label]
            print(label)
            row = i // 20
            col = i % 20
            print(i, col)
            print(row, col)
            combined[row*32: (row+1)*32, col*32: (col+1)*32] = img
            i += 1
            res += label
        cv2.imshow('rr', combined)
        print(res)
        cv2.imwrite('assets/combined.png', combined)
        cv2.waitKey(0)
            # break
    else:
        i = 0
        for data in val_ds.take(36):
            # start training on model...
            img, label = data['image'], data['label']
            img = img.numpy()
            img = np.array(img, dtype=np.uint8)
            print(img.shape)
            # img = cv2.resize(img, (64, 64))
            label = label.numpy()
            label = charactors[label]
            print(label)
            cv2.imshow('rr', img)
            cv2.imwrite('assets/{}.png'.format(i), img)
            i += 1
            cv2.waitKey(0)
            # break
