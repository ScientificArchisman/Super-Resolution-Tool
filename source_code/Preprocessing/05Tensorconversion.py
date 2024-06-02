import os
from PIL import Image
from torchvision import transforms
import torch

# Define paths
low_res_dir = '/Users/archismanchakraborti/Desktop/python_files/Super-Resolution-Tool/data/low_res_data'
high_res_dir = '/Users/archismanchakraborti/Desktop/python_files/Super-Resolution-Tool/data/high_res_data'
processed_low_res_dir = '/Users/archismanchakraborti/Desktop/python_files/Super-Resolution-Tool/data/processed_low_res'
processed_high_res_dir = '/Users/archismanchakraborti/Desktop/python_files/Super-Resolution-Tool/data/processed_high_res'

os.makedirs(processed_low_res_dir, exist_ok=True)
os.makedirs(processed_high_res_dir, exist_ok=True)

# Define transformations
lr_size = (32, 32)
hr_size = (128, 128)
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Process and save images
for idx, img_name in enumerate(sorted(os.listdir(low_res_dir))):

    if idx % 100 == 0:
        print(f"Processed {idx} images")
        
    low_res_path = os.path.join(low_res_dir, img_name)
    high_res_path = os.path.join(high_res_dir, img_name.replace('_lr', ''))
    
    low_res_image = Image.open(low_res_path).convert("RGB").resize(lr_size)
    high_res_image = Image.open(high_res_path).convert("RGB").resize(hr_size)
    
    low_res_tensor = transform(low_res_image)
    high_res_tensor = transform(high_res_image)
    
    # Save the transformed tensors
    img_number = img_name.split('_')[0]
    torch.save(low_res_tensor, os.path.join(processed_low_res_dir, f"{img_number}_lr.pt"))
    torch.save(high_res_tensor, os.path.join(processed_high_res_dir, f"{img_number}_hr.pt"))