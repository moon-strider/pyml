import numpy as np

from PIL import Image


def is_grey_scale(img) -> bool:
    if len(img.getbands()) == 1:
        return True
    elif len(set(pixel for pixel in img.getdata())) == 1:
        return True
    else:
        return False
    

def get_pixel(img, x, y, c=-1) -> float:
    w, _ = img.size
    if is_grey_scale(img) and c == -1:
        return img.getdata()[x + y * w][0]
    elif c in range(3):
        return img.getdata()[x + y * w][c]
    raise ValueError("You must provide a valid channel argument: [0, 1, 2] for [R, G, B].")\
        if c!=-1 else ValueError("The image must be greyscale if you do not provide a channel.")
    

def get_center_2d(mat) -> tuple: # todo: 1d, 3d
    pass