import numpy as np

from core.utils import ensure_1d

from PIL import Image


def is_grey_scale(img: Image) -> bool:
    if len(img.getbands()) == 1:
        return True
    elif len(set(pixel for pixel in img.getdata())) == 1:
        return True
    else:
        return False
    

def get_pixel(img: Image, x: int, y: int, c=-1) -> float:
    w, _ = img.size
    if is_grey_scale(img) and c == -1:
        return img.getdata()[x + y * w][0]
    elif c in range(3):
        return img.getdata()[x + y * w][c]
    raise ValueError("You must provide a valid channel argument: [0, 1, 2] for [R, G, B].")\
        if c!=-1 else ValueError("The image must be greyscale if you do not provide a channel.")
    

def get_center_2d(shape: tuple[int, int]) -> tuple[int, int]:
    """
    Get the central index of a 2d matrix. 
    If it does not have a single central point, returns the top-left one.
    
    :param tuple[int, int] shape: a row-major shape of a matrix
    """
    if len(shape) != 2:
        raise ValueError("You must provide a (w, h) tuple to use this function.")
    rows, cols = shape
    rows_center, cols_center = rows // 2, cols // 2
    if rows % 2 == 0 and cols % 2 == 0:
        return (rows_center - 1, cols_center - 1)
    elif rows % 2 == 0:
        return (rows_center - 1, cols_center)
    elif cols % 2 == 0:
        return (rows_center, cols_center - 1)
    else:
        return (rows_center, cols_center)
    

def get_center_1d(vec: np.ndarray) -> int: # TODO: accept n, not the vector itself?
    # if a vector does not have one central point, returns left-most one
    if vec.ndim != 1:
        raise ValueError("You must provide a 1D vector in this function.")
    n = vec.size
    if n % 2 == 0:
        return n // 2 - 1
    return n // 2
