import numpy as np

from PIL import Image


def is_grey_scale(img) -> bool:
    if len(img.getbands()) == 1:
        return True
    elif len(set(pixel for pixel in img.getdata())) == 1:
        return True
    else:
        return False