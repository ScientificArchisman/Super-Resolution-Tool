import torch


MODEL_WEIGHTS_PATH = "/Users/archismanchakraborti/Desktop/python_files/Super-Resolution-Tool/best_model_weights/ESRGAN/generator2.pth"
MODEL_INPUT_PARAMS = (3, 3, 64, 23, 32)
IMG_INPUT_SHAPE = (256, 256)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")