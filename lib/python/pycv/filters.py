#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np


# Log output setting.
# If handler = StreamHandler(), log will output into StandardOutput.
from logging import getLogger, NullHandler, StreamHandler, DEBUG
logger = getLogger(__name__)
handler = NullHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)

class Filter:
    # High-pass filter
    @classmethod
    def HPF(self,im_source,im_mask):
        if len(im_source.shape) == 3:
            return None
        im_float32 = np.float32(im_source)
        dft = cv2.dft(im_float32, flags = cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        fshift = dft_shift*im_mask
        f_ishift = np.fft.ifftshift(fshift)
        im_back = cv2.idft(f_ishift)
        im_back = cv2.magnitude(im_back[:,:,0],im_back[:,:,1])
        Pmax = np.max(im_back)
        im_pow = im_back/Pmax*255
        return np.uint8(im_pow)

    @classmethod
    def colorFilter(self,im_in,rgb, lab_dist):

        bgr_img = np.array([[np.flipud(rgb)]], dtype=np.uint8)
        lab_arr = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2Lab)[0,0].astype(np.int32)
        # lab_mat = np.full(im_in.shape, lab_arr);

        im_lab = cv2.cvtColor(im_in, cv2.COLOR_BGR2Lab).astype(np.int32)

        im_mask = np.linalg.norm(im_lab-lab_arr, axis=2).astype(np.float32)

        im_mask = cv2.threshold(im_mask, lab_dist, 255, cv2.THRESH_BINARY_INV)[1].astype(np.uint8)

        im_dst = cv2.bitwise_and(im_in, im_in, mask=im_mask)

        return im_dst
def colorFilter(im_in, rgb, dist):
    im_mask = np.linalg.norm(im_in.astype(np.int32) - np.flipud(rgb), axis=2).astype(np.float32)
    im_mask = cv2.threshold(im_mask, dist, 255, cv2.THRESH_BINARY_INV)[1].astype(np.uint8)
    im_dst = cv2.bitwise_and(im_in, im_in, mask=im_mask)
    return im_dst


def main(argv):
    import os
    path = os.path.abspath(os.path.dirname(__file__) )
    img_fn = os.path.join(path,"../../data","color_filter_test.png")
    im_in = cv2.imread(img_fn)
    # print(im_in)

    im_hsv = colorFilter(im_in,[0,255,0],100)
    cv2.imshow("AA",im_hsv)
    cv2.waitKey(0)
    print(img_fn)

    
    img = scipy.misc.lena()
    rows, cols = img.shape
    crow, ccol = rows/2 , cols/2
    mask = np.zeros((rows, cols, 1), np.uint8)
    cv2.circle(mask,(crow,ccol),radius = 50,color = 255,thickness = -1)
    
    mask = 255-mask
    im = Filter.HPF(img,mask)
    cv2.imshow("A",im)
    cv2.waitKey(0)

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
