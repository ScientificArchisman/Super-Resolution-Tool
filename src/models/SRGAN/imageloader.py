import os
from PIL import Image
import torch
torch.set_num_threads(1)
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import config


class PreprocessedDataset(Dataset):
    def __init__(self, processed_low_res_dir, processed_high_res_dir):
        self.processed_low_res_dir = processed_low_res_dir
        self.processed_high_res_dir = processed_high_res_dir
        self.low_res_images = sorted(os.listdir(processed_low_res_dir))
        self.high_res_images = sorted(os.listdir(processed_high_res_dir))

    def __len__(self):
        return len(self.low_res_images)

    def __getitem__(self, idx):
        low_res_image_name = self.low_res_images[idx]
        high_res_image_name = self.high_res_images[idx]
        
        low_res_path = os.path.join(self.processed_low_res_dir, low_res_image_name)
        high_res_path = os.path.join(self.processed_high_res_dir, high_res_image_name)
        
        low_res_tensor = torch.load(low_res_path)
        high_res_tensor = torch.load(high_res_path)
        
        return low_res_tensor, high_res_tensor

def create_dataloaders(low_res_dir, high_res_dir, batch_size=config.BATCH_SIZE, 
                       num_workers=config.NUM_WORKERS):
    # Create dataset
    dataset = PreprocessedDataset(low_res_dir, high_res_dir)

    # Split dataset into training (80%) and testing (20%)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Further split the training dataset into training (80%) and validation (20%)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

if __name__ == '__main__':

    train_loader, val_loader, test_loader = create_dataloaders(low_res_dir=config.LOW_RES_FOLDER,
                                                               high_res_dir=config.HIGH_RES_FOLDER,
                                                               batch_size=config.BATCH_SIZE, 
                                                               num_workers=config.NUM_WORKERS)

    for lr, hr in train_loader:
        print(f"Low-res batch shape: {lr.shape}")
        print(f"High-res batch shape: {hr.shape}")
        break

    for lr, hr in val_loader:
        print(f"Low-res batch shape: {lr.shape}")
        print(f"High-res batch shape: {hr.shape}")
        break

    for lr, hr in test_loader:
        print(f"Low-res batch shape: {lr.shape}")
        print(f"High-res batch shape: {hr.shape}")
        break


