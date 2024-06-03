'''
Digital Signal Processing Library for the webpage

'''

import cv2
import numpy as np

BIT = 8
L = 2**8

################# Filter #################
def GaussianFilter(image, kernel_size=5, sigma=1.0):
    kernel_size = int(kernel_size) // 2
    x, y = np.mgrid[-kernel_size:kernel_size+1, -kernel_size:kernel_size+1]
    g = np.exp(-(x**2 + y**2) / (2*sigma**2))
    g = g / g.sum()
    
    filtered_image = cv2.filter2D(image, -1, g)
    return filtered_image

################# Algorithm #################
def ErrorDiffusion(img, threshold=128):
    img_dither = img.astype(float).copy()
    h, w = img.shape[:2]
    channels = 1 if len(img.shape) == 2 else img.shape[2]
    
    # Create a padded array to handle edge cases
    padded_img = np.zeros((h + 1, w + 1, channels if channels > 1 else 1), dtype=float)
    padded_img[:h, :w] = img_dither if channels > 1 else img_dither[:, :, np.newaxis]

    for i in range(h):
        for j in range(w):
            for c in range(channels):
                old_pix = padded_img[i, j, c] if channels > 1 else padded_img[i, j, 0]
                
                # Quantization (Binarize)
                new_pix = 255 if old_pix > threshold else 0
                
                if channels == 1:
                    padded_img[i, j, 0] = new_pix
                else:
                    padded_img[i, j, c] = new_pix
                
                quant_err = old_pix - new_pix
                
                # Error diffusion to neighboring pixels
                if j + 1 < w:
                    padded_img[i, j + 1, c] += quant_err * 7 / 16
                if i + 1 < h:
                    padded_img[i + 1, j, c] += quant_err * 5 / 16
                    if j > 0:
                        padded_img[i + 1, j - 1, c] += quant_err * 3 / 16
                    if j + 1 < w:
                        padded_img[i + 1, j + 1, c] += quant_err * 1 / 16

    # Clip values to ensure they remain valid and convert back to uint8
    img_dither = np.clip(padded_img[:h, :w], 0, 255).astype(np.uint8)
    
    return img_dither if channels > 1 else img_dither[:, :, 0]
