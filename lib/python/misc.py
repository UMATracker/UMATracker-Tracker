import sys

import cv2
from PyQt5.QtGui import QImage


# Log output setting.
# If handler = StreamHandler(), log will output into StandardOutput.
from logging import getLogger, NullHandler, StreamHandler, DEBUG
logger = getLogger(__name__)
handler = NullHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)


systemEncoding = sys.stdout.encoding

def utfToSystemStr(utfStr):
    return utfStr.encode(systemEncoding)

def cvMatToQImage(im_in):
    logger.debug('Input image type: {0}'.format(im_in.dtype))
    if len(im_in.shape) is 3:
        height, width, bytesPerComponent = im_in.shape
        bytesPerLine = bytesPerComponent * width;
        im_dst = cv2.cvtColor(im_in, cv2.COLOR_BGR2RGB)
        return QImage(im_dst.data, width, height, bytesPerLine, QImage.Format_RGB888)
    else:
        height, width = im_in.shape
        return QImage(im_in.data, width, height, width, QImage.Format_Indexed8)
