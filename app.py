# app.py
from flask import Flask, request, render_template, send_file, jsonify
import os
from source_code.models.ESRGAN.model import RRDBNet
import torch
import numpy as np
from source_code.models.ESRGAN.preprocess import preprocess_image
import source_code.models.ESRGAN.config_ESR as config
from PIL import Image
import io
import cv2
import base64



app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image']
    model_choice = request.form['model']
    width = int(request.form['width'])
    height = int(request.form['height'])
    input_shape = (width, height)

    # Set model weights path based on the selected model
    model_weights_path = get_model_weights_path(model_choice)
    output_image = process_image(image, model_weights_path, input_shape)
    
    img_io = io.BytesIO()
    output_image.save(img_io, 'PNG')
    img_io.seek(0)
    img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')

    return jsonify({'image': img_base64})

def get_model_weights_path(model_choice):
    # Dummy paths, replace with your actual model weights paths
    if model_choice == 'SRGAN':
        return config.SRGAN_WEIGHTS_PATH
    elif model_choice == 'ESRGAN':
        return config.ESRGAN_WEIGHTS_PATH
    else:
        return config.DEFAULT_WEIGHTS_PATH

def process_image(image, model_weights_path, input_shape):
    model_input_params = config.MODEL_INPUT_PARAMS
    device = config.DEVICE
    
    # Read image using OpenCV
    image = np.frombuffer(image.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    
    # Initialize the model
    model = RRDBNet(*model_input_params).to(device)
    model.load_state_dict(torch.load(model_weights_path))
    model.eval()
    
    # Preprocess image
    img_LR = preprocess_image(image, input_shape)
    img_LR = img_LR.to(device)
    
    # Generate image
    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output_image = Image.fromarray((output * 255).astype(np.uint8))
    
    return output_image

if __name__ == '__main__':
    app.run(debug=True)
