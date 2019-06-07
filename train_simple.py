"""

training a simple net on Chinese Characters classification dataset
we got about 90% accuracy by simply applying a simple CNN net

"""
from alfred.dl.tf.common import mute_tf
mute_tf()
import os
import sys
import numpy as np
import tensorflow as tf

from alfred.utils.log import logger as logging
import tensorflow_datasets as tfds
from dataset.casia_hwdb import load_ds, load_characters, load_val_ds
from models.cnn_net import CNNNet, build_net_002, build_net_003



target_size = 64
num_classes = 7356
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


def train():
    all_characters = load_characters()
    num_classes = len(all_characters)
    logging.info('all characters: {}'.format(num_classes))
    train_dataset = load_ds()
    train_dataset = train_dataset.shuffle(100).map(preprocess).batch(32).repeat()

    val_ds = load_val_ds()
    val_ds = val_ds.shuffle(100).map(preprocess).batch(32).repeat()

    for data in train_dataset.take(2):
        print(data)

    # init model
    model = build_net_003((64, 64, 1), num_classes)
    model.summary()
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
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy'])
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(ckpt_path,
                                               save_weights_only=True,
                                               verbose=1,
                                               period=500)
        ]
        try:
            model.fit(
                train_dataset,
                validation_data=val_ds,
                validation_steps=1000,
                epochs=15000,
                steps_per_epoch=1024,
                callbacks=callbacks)
        except KeyboardInterrupt:
            model.save_weights(ckpt_path.format(epoch=0))
            logging.info('keras model saved.')
        model.save_weights(ckpt_path.format(epoch=0))
        model.save(os.path.join(os.path.dirname(ckpt_path), 'cn_ocr.h5'))
    else:
        loss_fn = tf.losses.SparseCategoricalCrossentropy()
        optimizer = tf.optimizers.RMSprop()

        train_loss = tf.metrics.Mean(name='train_loss')
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
