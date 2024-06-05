import cv2 
import numpy as np 
import torch

def preprocess_image(img: np.ndarray, input_shape: tuple):
    # Normalize image
    img = img / 255.0
    # Resize image
    img = cv2.resize(img, input_shape, interpolation=cv2.INTER_AREA)
    # Convert to tensor
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    # Add dimension
    img = img.unsqueeze(0)
    return img
