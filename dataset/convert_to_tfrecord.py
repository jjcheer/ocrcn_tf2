"""
generates HWDB data into tfrecord
"""
import struct
import numpy as np
import cv2
from alfred.utils.log import logger as logging
import tensorflow as tf
import glob


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
                sample_size = header[0] + (header[1]<<8) + (header[2]<<16) + (header[3]<<24)
                tagcode = header[5] + (header[4]<<8)
                width = header[6] + (header[7]<<8)
                height = header[8] + (header[9]<<8)
                if header_size + width*height != sample_size:
                    break
                image = np.fromfile(f, dtype='uint8', count=width*height).reshape((height, width))
                yield image, tagcode


def run():
    all_hwdb_gnt_files = glob.glob('./hwdb_raw/HWDB1.1trn_gnt/*.gnt')
    logging.info('got all {} gnt files.'.format(len(all_hwdb_gnt_files)))
    logging.info('gathering charset...')
    charset = []
    for gnt in all_hwdb_gnt_files:
        hwdb = CASIAHWDBGNT(gnt)
        for img, tagcode in hwdb.get_data_iter():
            try:
                label = struct.pack('>H', tagcode).decode('gb2312')            
                label = label.replace('\x00', '')
                charset.append(label)
            except Exception as e:
                continue
    charset = sorted(set(charset))
    logging.info('all got {} charactors.'.format(len(charset)))
    with open('charactors.txt', 'w') as f:
        f.writelines('\n'.join(charset))

    tfrecord_f = 'casia_hwdb_1.0_1.1.tfrecord'
    i = 0
    with tf.io.TFRecordWriter(tfrecord_f) as tfrecord_writer:
        for gnt in all_hwdb_gnt_files:
            hwdb = CASIAHWDBGNT(gnt)
            for img, tagcode in hwdb.get_data_iter():
                try:
                    img = cv.resize(img, (64, 64))
                    label = struct.pack('>H', tagcode).decode('gb2312')            
                    label = label.replace('\x00', '')
                    index = charset.index(label)
                    # save img, label as example
                    example = tf.train.Example(features=tf.train.Features(
                        feature={
                        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img]))
                    }))
                    tfrecord_writer.write(example.SerializeToString())
                    if i%500:
                        logging.info('solved {} examples.'.format(i))
                    i += 1
                except Exception as e:
                    logging.error(e)
                    continue
    logging.info('done.')

if __name__ == "__main__":
    run()