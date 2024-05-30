import os
from PIL import Image
import torchvision.transforms as transforms

# Define the transformations
def apply_gaussian_blur_and_downsample(image, downsample_factor):
    """Apply Gaussian blur and then downsample the image by the given factor."""
    # Apply Gaussian blur
    gaussian_blur = transforms.GaussianBlur(kernel_size=(5, 5), sigma=(1.0, 2.0))
    blurred_image = gaussian_blur(image)
    
    # Downsample the image
    width, height = blurred_image.size
    new_width, new_height = width // downsample_factor, height // downsample_factor
    downsample = transforms.Resize((new_height, new_width))
    low_res_image = downsample(blurred_image)
    
    return low_res_image

def load_image(image_path):
    """Load an image from the specified path."""
    return Image.open(image_path)

def save_image(image, image_path):
    """Save an image with the correct extension."""
    image.save(image_path)

def process_images(source_directory, target_directory, downsample_factor):
    """Load, process (blur and downsample), and save the images."""
    # Get a list of all files in the source directory
    files = os.listdir(source_directory)
    
    # Filter the list to include only image files with the desired extensions
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Sort the files to ensure they are processed in a consistent order
    image_files.sort()
    
    for file_name in image_files:
        # Create the full paths for the old and new file names
        old_file_path = os.path.join(source_directory, file_name)
        
        # Load, process, and save the image
        image = load_image(old_file_path)
        low_res_image = apply_gaussian_blur_and_downsample(image, downsample_factor)
        
        # Create the new filename with "lr" added at the end
        base_name, file_extension = os.path.splitext(file_name)
        new_file_name = f"{base_name}_lr{file_extension}"
        new_file_path = os.path.join(target_directory, new_file_name)
        
        save_image(low_res_image, new_file_path)
    
    print("Processing completed.")

if __name__ == "__main__":
    # Define the source and target directories
    source_directory = "/Users/archismanchakraborti/Desktop/python_files/GAN_TEST/data/high_res_data"
    target_directory = "/Users/archismanchakraborti/Desktop/python_files/GAN_TEST/data/low_res_data"
    
    # Create the target directory if it does not exist
    os.makedirs(target_directory, exist_ok=True)
    
    # Define the downsample factor
    downsample_factor = 4  # Adjust the downsample factor as needed
    
    # Process the images in the source directory and save them to the target directory
    process_images(source_directory, target_directory, downsample_factor)
