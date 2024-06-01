from torchvision import transforms
import torch


####### Discriminator config ######
DISCRIMINATOR_INPUT_CHANNELS = 3

###### Generator config ######
GENERATOR_INPUT_CHANNELS = 3
GENERATOR_OUTPUT_CHANNELS = 3
NUM_UPSAMPLE_BLOCKS = 2
NUM_RESIDUAL_BLOCKS = 16



############## Dataloader config ##############
LOW_RES_FOLDER = "/Users/archismanchakraborti/Desktop/python_files/Super-Resolution-Tool/data/processed_low_res"
HIGH_RES_FOLDER = "/Users/archismanchakraborti/Desktop/python_files/Super-Resolution-Tool/data/processed_high_res"
TRANSFORMS = transforms.Compose([
    transforms.ToTensor()
])

############ Training config ############
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_EPOCHS = 100
BATCH_SIZE = 16
NUM_WORKERS = 0

