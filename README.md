# SuperSampling Project with SRGAN and ESRGAN

This project aims to develop a Super-Resolution tool using advanced Generative Adversarial Networks (GANs) such as SRGAN and ESRGAN. The project involves preprocessing a dataset, training models, and deploying the final model as a tool for enhancing image resolution.

## Dataset

We are using the DIV2K dataset, which is commonly used for image super-resolution tasks. The dataset initially contains 800 high-resolution images.

## Preprocessing Steps

### 1. Image Splitting

The original 800 images were divided into 16 sub-images each, resulting in a more extensive dataset for training. This helps in improving the model's performance by providing more diverse data points.

### 2. Data Augmentation

To further enhance the dataset, we applied various data augmentation techniques. This included:

- **Random Horizontal and Vertical Flips:** Flipping images horizontally and vertically to increase variability.
- **Random Rotation:** Rotating images by random angles.
- **Color Jittering:** Adjusting the brightness, contrast, saturation, and hue of the images.
- **Random Resized Cropping:** Randomly cropping and resizing images to introduce scale variations.

### 3. Creating Low-Resolution Images

To generate low-resolution images from the augmented data, we applied Gaussian blur followed by downsampling. This process involved:

- **Gaussian Blur:** Applying a Gaussian blur to each image to simulate real-world degradation.
- **Downsampling:** Reducing the image resolution by a factor to create the low-resolution dataset.

These low-resolution images will serve as the input data for training our Super-Resolution models.

## Current Stage

We are currently at the preprocessing stage, where the high-resolution images from the DIV2K dataset have been split, augmented, and downsampled to create the low-resolution training dataset.

## Next Steps

1. **Model Training:** We will train the Super-Resolution models (SRGAN and ESRGAN) using the preprocessed data.
2. **Model Evaluation:** Evaluate the performance of the trained models on validation and test sets.
3. **Deployment:** Deploy the best-performing model as a tool for enhancing image resolution.

## Project Structure

├── data
│ ├── original_data # Original high-resolution images
│ ├── augmented_data # Augmented high-resolution images
│ ├── low_res_data # Low-resolution images created after Gaussian blur and downsampling
├── src
│ ├── preprocessing.py # Script for preprocessing the data
│ ├── training.py # Script for training the SRGAN and ESRGAN models
│ ├── evaluation.py # Script for evaluating the models
│ ├── deployment.py # Script for deploying the trained model as a tool
└── README.md # Project description and instructions


## Requirements

- Python 3.x
- PyTorch
- torchvision
- PIL
- numpy
- matplotlib

## How to Run

1. Clone this repository.
2. Install the required packages using `pip install -r requirements.txt`.
3. Run the preprocessing script to prepare the data:
   ```bash
   python src/preprocessing.py

Stay tuned for more updates as we progress through the training and deployment stages!
## Acknowledgements

    - The DIV2K dataset for providing high-quality images for our project.
    - The PyTorch community for their excellent machine learning library and resources.

