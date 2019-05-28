"""

this is a wrapper handle CASIA_HWDB dataset
since original data is complicated
we using this class to get .png and label from raw
.gnt data

"""
import struct
import numpy as np
import cv2


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


def resize_padding_or_crop(target_size, ori_img, padding_value=255):
    if len(ori_img.shape) == 3:
        res = np.zeros([ori_img.shape[0], target_size, target_size])
    else:
        res = np.ones([target_size, target_size])*padding_value
        end_x = target_size
        end_y = target_size
        start_x = 0
        start_y = 0
        if ori_img.shape[0] < target_size:
            end_x = int((target_size + ori_img.shape[0])/2)
        if ori_img.shape[1] < target_size:
            end_y = int((target_size + ori_img.shape[1])/2)
        if ori_img.shape[0] < target_size:
            start_x = int((target_size - ori_img.shape[0])/2)
        if ori_img.shape[1] < target_size:
            start_y = int((target_size - ori_img.shape[1])/2)
        res[start_x:end_x, start_y:end_y] = ori_img
        return np.array(res, dtype=np.uint8)

if __name__ == "__main__":
    gnt = CASIAHWDBGNT('samples/1001-f.gnt')

    full_img = np.zeros([900, 900], dtype=np.uint8)
    charset = []
    i = 0
    for img, tagcode in gnt.get_data_iter():
        # cv2.imshow('rr', img)
        try:
            label = struct.pack('>H', tagcode).decode('gb2312')            
            img_padded = resize_padding_or_crop(90, img)
            col_idx = i%10
            row_idx = i//10
            full_img[row_idx*90:(row_idx+1)*90, col_idx*90:(col_idx+1)*90] = img_padded
            charset.append(label.replace('\x00', ''))
            if i >= 99:
                cv2.imshow('rrr', full_img)
                cv2.imwrite('sample.png', full_img)
                cv2.waitKey(0)
                print(charset)
                break
            i += 1
        except Exception as e:
            # print(e.with_traceback(0))
            print('decode error')
            continue
