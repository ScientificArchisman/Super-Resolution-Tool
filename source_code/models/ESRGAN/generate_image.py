from model import RRDBNet
import torch
import numpy as np
from preprocess import preprocess_image
import config_ESR as config

def generate(image_lr, model_weights_path = config.MODEL_WEIGHTS_PATH, 
             model_input_params = config.MODEL_INPUT_PARAMS, 
             img_input_shape = config.IMG_INPUT_SHAPE, 
             device = config.DEVICE):
    # Load model
    model = RRDBNet(*model_input_params).to(device)
    model.load_state_dict(torch.load(model_weights_path))
    model.eval()
    
    # Preprocess image
    img_LR = preprocess_image(image_lr, img_input_shape)
    img_LR.to(device)
    
    # Generate image
    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    return output