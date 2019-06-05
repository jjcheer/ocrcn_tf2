'''
training HWDB Chinese charactors classification 
on MobileNetV2 
'''
from alfred.dl.tf.common import mute_tf
mute_tf()

import os
import sys
import numpy as np
import tensorflow as tf

from alfred.utils.log import logger as logging
import tensorflow_datasets as tfds
from dataset.casia_hwdb import load_ds, load_charactors
from models.cnn_net import CNNNet


target_size = 224
num_classes = 7356
use_keras_fit = False
# use_keras_fit = True
ckpt_path = './checkpoints/no_finetune/flowers_mbv2_scratch-{epoch}.ckpt'


def preprocess(x):
    """
    minus mean pixel or normalize?
    """
    x['image'] = tf.image.resize(x['image'], (target_size, target_size))
    x['image'] /= 255.
    x['image'] = 2*x['image'] - 1
    return x['image'], x['label']

def train():
    all_charactors = load_charactors()
    num_classes = len(all_charactors)
    # using mobilenetv2 classify tf_flowers dataset
    train_dataset = load_ds()
    train_dataset = train_dataset.shuffle(100).map(preprocess).batch(4).repeat()

    # init model
    model = CNNNet()

    # model.summary()
    # model = tf.keras.models.load_model('flowers_mobilenetv2.h5')
    logging.info('model loaded.')
    
    start_epoch = 0
    latest_ckpt = tf.train.latest_checkpoint(os.path.dirname(ckpt_path))
    if latest_ckpt:
        start_epoch = int(latest_ckpt.split('-')[1].split('.')[0])
        model.load_weights(latest_ckpt)
        logging.info('model resumed from: {}, start at epoch: {}'.format(latest_ckpt, start_epoch))
    else:
        logging.info('passing resume since weights not there. training from scratch')

    if use_keras_fit:
        # todo: why keras fit converge faster than tf loop?
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
        try:
            model.fit(
            train_dataset, epochs=50,             
            steps_per_epoch=700,)
        except KeyboardInterrupt:
            model.save_weights(ckpt_path.format(epoch=0))
            logging.info('keras model saved.')
        model.save_weights(ckpt_path.format(epoch=0))
        model.save(os.path.join(os.path.dirname(ckpt_path), 'flowers_mobilenetv2.h5'))
    else:
        loss_fn = tf.losses.SparseCategoricalCrossentropy()
        optimizer = tf.optimizers.RMSprop()

        train_loss = tf.metrics.Mean(name='train_loss')
        # the accuracy calculation has some problems, seems not right?
        train_accuracy = tf.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        for epoch in range(start_epoch, 120):
            try:
                for batch, data in enumerate(train_dataset):
                    # images, labels = data['image'], data['label']
                    images, labels = data
                    with tf.GradientTape() as tape:
                        predictions = model(images)
                        loss = loss_fn(labels, predictions)
                    gradients = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                    train_loss(loss)
                    train_accuracy(labels, predictions)
                    if batch % 10 == 0:
                        logging.info('Epoch: {}, iter: {}, loss: {}, train_acc: {}'.format(
                        epoch, batch, train_loss.result(), train_accuracy.result()))
            except KeyboardInterrupt:
                logging.info('interrupted.')
                model.save_weights(ckpt_path.format(epoch=epoch))
                logging.info('model saved into: {}'.format(ckpt_path.format(epoch=epoch)))
                exit(0)



if __name__ == "__main__":
    train()

