import logging
import os
from datetime import datetime

def setup_logging(log_dir, model_name, config):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create loggers
    batch_logger = logging.getLogger('batch_logger')
    epoch_logger = logging.getLogger('epoch_logger')
    
    # Set level
    batch_logger.setLevel(logging.INFO)
    epoch_logger.setLevel(logging.INFO)
    
    # Create file handlers
    batch_log_path = os.path.join(log_dir, 'batch.log')
    epoch_log_path = os.path.join(log_dir, 'epoch.log')
    
    batch_file_handler = logging.FileHandler(batch_log_path)
    epoch_file_handler = logging.FileHandler(epoch_log_path)
    
    # Create formatter for logs with timestamps
    log_formatter = logging.Formatter('%(asctime)s - %(message)s')
    
    # Add formatter to handlers
    batch_file_handler.setFormatter(log_formatter)
    epoch_file_handler.setFormatter(log_formatter)
    
    # Add handlers to loggers
    batch_logger.addHandler(batch_file_handler)
    epoch_logger.addHandler(epoch_file_handler)
    
    # Log model name and configuration without timestamps
    batch_logger.propagate = False
    epoch_logger.propagate = False

    with open(batch_log_path, 'a') as f:
        f.write(f"Model: {model_name}\n")
        f.write("Configuration:\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
        f.write('-' * 50 + '\n')
        f.write('Epoch: , Batch: , Generator Loss: , Discriminator Loss: , Content Loss: , Adversarial Loss: \n')
    
    with open(epoch_log_path, 'a') as f:
        f.write(f"Model: {model_name}\n")
        f.write("Configuration:\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
        f.write('-' * 50 + '\n')
        f.write('Epoch: , Avg Generator Loss: , Avg Discriminator Loss: , Avg Content Loss: , Avg Adversarial Loss: , Avg Validation Generator Loss: , Avg Validation Discriminator Loss: \n')
    
    return batch_logger, epoch_logger
