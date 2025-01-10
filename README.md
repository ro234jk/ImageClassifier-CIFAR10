# ImageClassifier-CIFAR10


This repository contains a Convolutional Neural Network (CNN) implementation for classifying images from the CIFAR-10 dataset. The model is based on the LeNet-5 architecture and achieves significant accuracy on the test dataset.

## Overview
The CIFAR-10 dataset is a collection of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. This project demonstrates the use of deep learning to classify these images accurately.

## Dataset
The CIFAR-10 dataset consists of the following classes:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

## Model Architecture
This project implements the LeNet-5 architecture:
1. **Convolutional Layer 1**: 6 filters, 5x5 kernel, tanh activation
2. **Average Pooling Layer 1**
3. **Convolutional Layer 2**: 16 filters, 5x5 kernel, tanh activation
4. **Average Pooling Layer 2**
5. **Convolutional Layer 3**: 120 filters, 5x5 kernel, tanh activation
6. **Flatten Layer**
7. **Fully Connected Layer 1**: 84 units, tanh activation
8. **Output Layer**: 10 units, softmax activation

## Dependencies
The project requires the following Python libraries:
- TensorFlow
- Matplotlib

You can install the dependencies using:
```bash
pip install tensorflow matplotlib
```

## Running the Project
1. Clone this repository:
   ```bash
   git clone <repository_url>
   ```
2. Navigate to the project directory:
   ```bash
   cd CIFAR10_CNN
   ```
3. Run the Python script:
   ```bash
   python cifar10_cnn.py
   ```

## Features
- **Data Preprocessing**: Normalization of images and one-hot encoding of labels.
- **Visualization**: Sample images from the dataset are visualized with their labels.
- **Training and Validation**: The model is trained for 10 epochs with validation at each step.
- **Evaluation**: Final accuracy on the test dataset is printed.
- **Plots**: Training and validation accuracy and loss are plotted for performance analysis.

## Results
After training for 10 epochs, the model achieves a test accuracy of approximately `~70-80%` (depending on hardware and configurations).

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Author
Adarsh Pal

Feel free to contribute or suggest improvements!
