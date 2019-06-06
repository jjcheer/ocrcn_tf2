'''



conv_1 = slim.conv2d(images, 64, [3, 3], 1, padding='SAME', scope='conv1')
# (inputs,num_outputs,[卷积核个数] kernel_size,[卷积核的高度，卷积核的宽]stride=1,padding='SAME',)
max_pool_1 = slim.max_pool2d(conv_1, [2, 2], [2, 2], padding='SAME')
conv_2 = slim.conv2d(max_pool_1, 128, [3, 3], padding='SAME', scope='conv2')
max_pool_2 = slim.max_pool2d(conv_2, [2, 2], [2, 2], padding='SAME')
conv_3 = slim.conv2d(max_pool_2, 256, [3, 3], padding='SAME', scope='conv3')
max_pool_3 = slim.max_pool2d(conv_3, [2, 2], [2, 2], padding='SAME')

flatten = slim.flatten(max_pool_3)
fc1 = slim.fully_connected(tf.nn.dropout(flatten, keep_prob), 1024, activation_fn=tf.nn.tanh, scope='fc1')
logits = slim.fully_connected(tf.nn.dropout(fc1, keep_prob), FLAGS.charset_size, activation_fn=None, scope='fc2')
# logits = slim.fully_connected(flatten, FLAGS.charset_size, activation_fn=None, reuse=reuse, scope='fc')
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
# y表示的是实际类别，y_表示预测结果，这实际上面是把原来的神经网络输出层的softmax和cross_entrop何在一起计算，为了追求速度
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), labels), tf.float32))
'''

import tensorflow as tf
from tensorflow.keras import layers


# some simple models
def build_net_001(input_shape, n_classes):
    assert len(input_shape) == 3, 'only support 3 channels'
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(
        input_shape=input_shape, filters=32, kernel_size=(3, 3), strides=(1, 1),
        padding='valid', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))
    return model


def build_net_002(input_shape, n_classes):
    model = tf.keras.Sequential([
        layers.Conv2D(input_shape=input_shape, filters=64, kernel_size=(3, 3), strides=(1, 1),
                      padding='same', activation='relu'),
        layers.MaxPool2D(pool_size=(2, 2), padding='same'),
        layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same'),
        layers.MaxPool2D(pool_size=(2, 2), padding='same'),
        layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same'),
        layers.MaxPool2D(pool_size=(2, 2), padding='same'),

        layers.Flatten(),
        layers.Dense(1024, activation='relu'),
        layers.Dense(n_classes, activation='softmax')
    ])
    return model


# some models wrapped into tf.keras.Model
class CNNNet(tf.keras.Model):

    def __init__(self):
        pass
