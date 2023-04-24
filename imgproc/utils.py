import numpy as np

from core.utils import ensure_1d

from PIL import Image


def is_grey_scale(img: Image) -> bool:
    """
    Chech whether an image is grey-scale or not.

    :param Image img: a PIL image to check
    :return: a boolean of image being grey-scale
    :rtype: bool
    """
    if len(img.getbands()) == 1:
        return True
    elif len(set(pixel for pixel in img.getdata())) == 1:
        return True
    else:
        return False
    

def get_pixel(img: Image, x: int, y: int, c=-1) -> float:
    """
    Get the pixel value in a PIL Image by providing x, y, and channel.

    :param Image img: a PIL image to get the pixel from
    :param int x: the x coordinate of a wanted pixel
    :param int y: the y coordinate of a wanted pixel
    :param int c: the channel of an image to get the pixel value from, defaults to -1
    :return: central matrix point index
    :rtype: tuple[int, int]
    :raises ValueError: if shape does not resemble a matrix
    """
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
    :return: central matrix point index
    :rtype: tuple[int, int]
    :raises ValueError: if shape does not resemble a matrix
    """
    if len(shape) != 2:
        raise ValueError("You must provide a (w, h) tuple in this function.")
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
    

def get_center_1d(n: int) -> int:
    """
    Get the central index of a 1d vector.
    If it does not have a single central point, returns the left-most one.

    :param int n: the size of a vector
    :return: central vector point index
    :rtype: int
    :raises ValueError: if the n parameter is not an integer
    """
    if not isinstance(n, int):
        raise ValueError("You must provide a 1D vector size in this function.")
    if n % 2 == 0:
        return n // 2 - 1
    return n // 2
