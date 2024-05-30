import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

# Define the augmentation transformations
augmentation_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomResizedCrop(size=(128, 128), scale=(0.8, 1.0)),
])

def load_image(image_path):
    """Load an image from the specified path."""
    return Image.open(image_path)

def save_augmented_image(image, image_path):
    """Save an augmented image with the correct extension."""
    image.save(image_path)

def augment_image(image_path, num_augmented_per_image=5):
    """Augment and save images."""
    image = load_image(image_path)
    base_name = os.path.basename(image_path).split('.')[0]
    augmented_images = []

    for i in range(num_augmented_per_image):
        augmented_image = augmentation_transforms(image)
        augmented_image_path = os.path.join(os.path.dirname(image_path), f"{base_name}_aug_{i}.png")
        augmented_images.append((augmented_image, augmented_image_path))
    
    return augmented_images

def process_and_save_augmented_images(image_path, num_augmented_per_image=5):
    """Process each image by augmenting it and saving the augmented versions."""
    augmented_images = augment_image(image_path, num_augmented_per_image)
    for augmented_image, augmented_image_path in augmented_images:
        save_augmented_image(augmented_image, augmented_image_path)

if __name__ == '__main__':
    BASE_DIRECTORY = "/Users/archismanchakraborti/Desktop/python_files/GAN_TEST/data/high_res_data"
    
    image_paths = [os.path.join(BASE_DIRECTORY, image_name) for image_name in os.listdir(BASE_DIRECTORY) if image_name.endswith(('.png', '.jpg', '.jpeg'))]

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_and_save_augmented_images, image_path) for image_path in image_paths]
        for idx, future in enumerate(futures):
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1} images")
    
    print("Data augmentation completed.")


