import os
import sys  


# Ensure the source_code directory is in the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'source_code'))

from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import numpy as np
from models.SRGAN.model import Generator, Discriminator
import models.SRGAN.config as config
from models.SRGAN.imageloader import create_dataloaders
from models.SRGAN.blocks import Generator_Residual_Block, Discriminator_Block, UpsampleBlock

generator = Generator(in_channels=config.GENERATOR_INPUT_CHANNELS, 
                      out_channels=config.GENERATOR_OUTPUT_CHANNELS, 
                      num_upsample_blocks=config.NUM_UPSAMPLE_BLOCKS, 
                      num_residual_blocks=config.NUM_RESIDUAL_BLOCKS).to(config.DEVICE)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Assuming you send data as JSON
    input_data = np.array(data['input']).astype(np.float32)  # Adjust based on your model's input shape
    input_tensor = torch.tensor(input_data).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        prediction = generator(input_tensor)
    result = prediction.cpu().numpy()  # Adjust based on your prediction logic
    return jsonify({'prediction': result.tolist()})

if __name__ == '__main__':
    app.run(debug=True)