import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from concurrent.futures import ProcessPoolExecutor, as_completed

def BreakImages(image, n_images: int) -> np.ndarray:
    """Split an image into n_images images using PyTorch for faster computation.
    Args:
        image: np.ndarray: The image to split
        n_images: int: The number of images to split the image into
    Returns:
        np.ndarray: The split images
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    image = torch.tensor(image).to(device)
    n_rows, n_cols, n_channels = image.shape
    n_images_sqrt = int(np.sqrt(n_images))

    if n_images_sqrt**2 != n_images:
        raise ValueError("n_images must be a perfect square")

    split_images = []
    for i in range(n_images_sqrt):
        for j in range(n_images_sqrt):
            split_image = image[
                i*n_rows//n_images_sqrt:(i+1)*n_rows//n_images_sqrt,
                j*n_cols//n_images_sqrt:(j+1)*n_cols//n_images_sqrt,
                :
            ]
            split_images.append(split_image.cpu().numpy())

    return np.array(split_images)

def save_image(image, image_path):
    """Save an image with the correct extension.
    Args:
        image: np.ndarray: The image to save
        image_path: str: The path to save the image to
    """
    if image_path.endswith((".png", ".jpg", ".jpeg")):
        plt.imsave(image_path, image)
    elif image_path.endswith(".npy"):
        np.save(image_path, image)
    else:
        raise ValueError("Image path must end with .png or .npy")

def process_image(image_path, saving_directory):
    """Process an image: read, split, and save the image."""
    image = plt.imread(image_path)
    image_name = os.path.basename(image_path)
    try:
        split_images = BreakImages(image, 16)
        for i, split_image in enumerate(split_images):
            save_image(split_image, os.path.join(saving_directory, f"{image_name.split('.')[0]}_{i}.png"))
    except Exception as e:
        print(f"Skipping {image_name} due to error: {e}")

if __name__ == '__main__':
    BASE_DIRECTORY = "/Users/archismanchakraborti/Desktop/python_files/GAN_TEST/data/original_data"
    SAVING_DIRECTORY = "/Users/archismanchakraborti/Desktop/python_files/GAN_TEST/data/high_res_data"
    os.makedirs(SAVING_DIRECTORY, exist_ok=True)

    image_paths = [os.path.join(BASE_DIRECTORY, image_name) for image_name in os.listdir(BASE_DIRECTORY) if image_name.endswith(('.png', '.jpg', '.jpeg'))]

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_image, image_path, SAVING_DIRECTORY) for image_path in image_paths]
        for idx, future in enumerate(as_completed(futures)):
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1} images")


