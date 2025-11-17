import os
import cv2
import numpy as np
from skimage import util


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def to_grayscale(img):
    if len(img.shape) == 2:
        return img.astype(np.uint8)
    # img is RGB
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return gray


def add_salt_pepper(img, amount=0.01):
    imgf = util.random_noise(img, mode='s&p', amount=amount)
    # random_noise returns float in [0,1]
    if imgf.dtype == np.float64 or imgf.max() <= 1.0:
        imgf = (imgf * 255).astype(np.uint8)
    return imgf


def add_gaussian_noise(img, var=0.01):
    imgf = util.random_noise(img, mode='gaussian', var=var)
    if imgf.dtype == np.float64 or imgf.max() <= 1.0:
        imgf = (imgf * 255).astype(np.uint8)
    return imgf


def histogram_equalization(img):
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    return cv2.equalizeHist(img)


def clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe_obj.apply(img)


def denoise_gaussian(img, ksize=3, sigma=0):
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def denoise_median(img, ksize=3):
    return cv2.medianBlur(img, ksize)


def denoise_bilateral(img, d=9, sigmaColor=75, sigmaSpace=75):
    return cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)


def save_image(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # convert RGB to BGR for OpenCV if needed
    if len(img.shape) == 3 and img.shape[2] == 3:
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(path, img)
