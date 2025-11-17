import cv2
import numpy as np
from scipy import ndimage


def sobel_edges(img):
    dx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.hypot(dx, dy)
    mag = (mag / np.max(mag) * 255).astype(np.uint8)
    return mag


def prewitt_edges(img):
    kx = np.array([[1,0,-1],[1,0,-1],[1,0,-1]], dtype=np.float32)
    ky = kx.T
    dx = ndimage.convolve(img.astype(np.float32), kx)
    dy = ndimage.convolve(img.astype(np.float32), ky)
    mag = np.hypot(dx, dy)
    mag = (mag / np.max(mag) * 255).astype(np.uint8)
    return mag


def log_edges(img, ksize=5):
    # LoG: Gaussian blur then Laplacian
    g = cv2.GaussianBlur(img, (ksize, ksize), 0)
    lap = cv2.Laplacian(g, cv2.CV_64F)
    mag = np.absolute(lap)
    mag = (mag / np.max(mag) * 255).astype(np.uint8)
    return mag


def canny_edges(img, sigma=0.33):
    # auto thresholds based on median
    v = np.median(img)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(img, lower, upper)
    return edges


def otsu_threshold(img):
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def apply_otsu_to_gradient(grad_img):
    return otsu_threshold(grad_img)


def mse(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return np.mean((a - b) ** 2)


def psnr(a, b, data_range=255.0):
    m = mse(a, b)
    if m == 0:
        return float('inf')
    return 10 * np.log10((data_range ** 2) / m)
