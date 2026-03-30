import cv2
import numpy as np
import pywt

def w2d(img, mode="haar", level=1):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = np.float32(gray) / 255.0
    coeffs = pywt.wavedec2(gray, mode, level=level)
    coeffs = list(coeffs)
    coeffs[0] *= 0
    reconstructed = pywt.waverec2(coeffs, mode)
    reconstructed = np.uint8(reconstructed * 255)
    return reconstructed
