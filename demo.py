"""

inference on a single Chinese character
image and recognition the meaning of it

"""
from alfred.dl.tf.common import mute_tf
mute_tf()
import os
import cv2
import sys
import numpy as np
import tensorflow as tf

from alfred.utils.log import logger as logging
import tensorflow_datasets as tfds
from dataset.casia_hwdb import load_ds, load_characters, load_val_ds
from models.cnn_net import CNNNet, build_net_002, build_net_003
import glob


target_size = 64
characters = load_characters()
num_classes = len(characters)
# use_keras_fit = False
use_keras_fit = True
ckpt_path = './checkpoints/cn_ocr-{epoch}.ckpt'


def preprocess(x):
    """
    minus mean pixel or normalize?
    """
    # original is 64x64, add a channel dim
    x['image'] = tf.expand_dims(x['image'], axis=-1)
    x['image'] = tf.image.resize(x['image'], (target_size, target_size))
    x['image'] = (x['image'] - 128.) / 128.
    return x['image'], x['label']


def get_model():
    # init model
    model = build_net_003((64, 64, 1), num_classes)
    logging.info('model loaded.')

    latest_ckpt = tf.train.latest_checkpoint(os.path.dirname(ckpt_path))
    if latest_ckpt:
        start_epoch = int(latest_ckpt.split('-')[1].split('.')[0])
        model.load_weights(latest_ckpt)
        logging.info('model resumed from: {} at epoch: {}'.format(latest_ckpt, start_epoch))
        return model
    else:
        logging.error('can not found any checkpoints matched: {}'.format(ckpt_path))


def predict(model, img_f):
    ori_img = cv2.imread(img_f)
    img = tf.expand_dims(ori_img[:, :, 0], axis=-1)
    img = tf.image.resize(img, (target_size, target_size))
    img = (img - 128.)/128.
    img = tf.expand_dims(img, axis=0)
    print(img.shape)
    out = model(img).numpy()
    print('predict: {}'.format(characters[np.argmax(out[0])]))
    cv2.imwrite('assets/pred_{}.png'.format(characters[np.argmax(out[0])]), ori_img)


if __name__ == '__main__':
    img_files = glob.glob('assets/*.png')
    model = get_model()
    for img_f in img_files:
        a = cv2.imread(img_f)
        cv2.imshow('rr', a)
        predict(model, img_f)
        cv2.waitKey(0)


