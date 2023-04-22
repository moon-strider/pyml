import numpy as np

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
    

def get_center_2d(mat: np.ndarray) -> np.ndarray:
    if mat.ndim != 2:
        raise ValueError("You must provide a 2D matrix in this function.")
    r, c = mat.shape
    rc, cc = r // 2, c // 2
    if r % 2 == 0 and c % 2 == 0:
        return np.array([(rc - 1, cc - 1), (rc - 1, cc), (rc, cc - 1), (rc, cc)])
    elif r % 2 == 0:
        return np.array([(rc - 1, cc), (rc, cc)])
    elif c % 2 == 0:
        return np.array([(rc, cc - 1), (rc, cc)])
    else:
        return np.array([(rc, cc)])
    

def get_center_1d(vec: np.ndarray) -> np.ndarray:
    if vec.ndim != 1:
        raise ValueError("You must provide a 1D vector in this function.")
    n = vec.size
    if n % 2 == 0:
        return np.array([n // 2 - 1, n // 2])
    return np.array([n // 2])