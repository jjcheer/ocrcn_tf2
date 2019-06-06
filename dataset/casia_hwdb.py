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


class CASIAHWDBGNT(object):
    """
    A .gnt file may contains many images and charactors
    """

    def __init__(self, f_p):
        self.f_p = f_p

    def get_data_iter(self):
        header_size = 10
        with open(self.f_p, 'rb') as f:
            while True:
                header = np.fromfile(f, dtype='uint8', count=header_size)
                if not header.size:
                    break
                sample_size = header[0] + (header[1] << 8) + (
                        header[2] << 16) + (header[3] << 24)
                tagcode = header[5] + (header[4] << 8)
                width = header[6] + (header[7] << 8)
                height = header[8] + (header[9] << 8)
                if header_size + width * height != sample_size:
                    break
                image = np.fromfile(f, dtype='uint8',
                                    count=width * height).reshape(
                    (height, width))
                yield image, tagcode


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
    label = tf.cast(features['label'], tf.int32)
    return {'image': img, 'label': label}


def load_ds():
    input_files = ['dataset/hwdb_11.tfrecord']
    ds = tf.data.TFRecordDataset(input_files)
    ds = ds.map(parse_example)
    return ds


def load_characters():

    a = open(os.path.join(this_dir, 'charactors.txt'), 'r').readlines()
    return [i.strip() for i in a]


if __name__ == "__main__":
    ds = load_ds()
    charactors = load_characters()
    for img, label in ds.take(9):
        # start training on model...
        img = img.numpy()
        img = np.resize(img, (64, 64))
        print(img.shape)
        label = label.numpy()
        label = charactors[label]
        print(label)
        cv2.imshow('rr', img)
        cv2.waitKey(0)
        # break
