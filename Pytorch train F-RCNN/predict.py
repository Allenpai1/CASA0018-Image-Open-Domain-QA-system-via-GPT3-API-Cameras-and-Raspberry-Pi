#----------------------------------------------------#
#   predict single image
#----------------------------------------------------#

import time
import numpy as np
from PIL import Image

from frcnn import FRCNN

if __name__ == "__main__":
    frcnn = FRCNN()
    mode = "predict"
    #-------------------------------------------------------------------------#
    #   crop                whether to crop image with bounding box detected
    #   count               to count the detection objects
    #-------------------------------------------------------------------------#
    crop            = True
    count           = False
    #-------------------------------------------------------------------------#
    #   dir_origin_path     input image path
    #   dir_save_path       save path
    #-------------------------------------------------------------------------#
    dir_origin_path = "img/"
    dir_save_path   = "img_out/"

    if mode == "predict":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                label, r_image = frcnn.detect_image(image, crop = crop, count = count)
                r_image.show()
