import os

def rename_images_in_directory(directory):
    # Get a list of all files in the directory
    files = os.listdir(directory)
    
    # Filter the list to include only image files with the desired extensions
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Sort the files to ensure they are processed in a consistent order
    image_files.sort()
    
    # Initialize a counter
    counter = 1
    
    # Loop through all image files and rename them
    for file_name in image_files:
        # Get the file extension
        file_extension = os.path.splitext(file_name)[1]
        
        # Create the new file name
        new_file_name = f"{counter}{file_extension}"
        
        # Create the full paths for the old and new file names
        old_file_path = os.path.join(directory, file_name)
        new_file_path = os.path.join(directory, new_file_name)
        
        # Rename the file
        os.rename(old_file_path, new_file_path)
        
        # Increment the counter
        counter += 1

    print("Renaming completed.")

if __name__ == "__main__":
    # Define the directory containing the images
    directory = "/Users/archismanchakraborti/Desktop/python_files/GAN_TEST/data/high_res_data"
    
    # Rename the images in the specified directory
    rename_images_in_directory(directory)
