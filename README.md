# Late Blight Detection in Potato Leaves using CNN

This project implements a Convolutional Neural Network (CNN) to detect Late Blight disease in potato leaves. Late Blight, caused by Phytophthora infestans, can severely reduce crop yield. Early detection using deep learning helps farmers take timely action.

## Aim

To build and train a CNN model capable of classifying potato leaves as Healthy or Late Blight infected with high accuracy.

## Features

* Dataset preprocessing and augmentation
* CNN model architecture for classification
* Training and validation with accuracy and loss visualization
* Prediction script for testing custom leaf images

## Execution Steps

1. Clone the repository

   ```bash
   git clone https://github.com/your-username/late-blight-detector.git
   cd late-blight-detector
   ```

2. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

3. Train the model

   ```bash
   python train.py
   ```

4. Predict using a test image

   ```bash
   python predict.py --image path_to_image.jpg
   ```

## Model Architecture

* Convolutional layers for feature extraction
* Pooling layers for dimensionality reduction
* Fully connected layers for classification
* Softmax activation for final prediction

## Applications

* Early detection of potato leaf diseases
* Precision agriculture for crop health monitoring
* Supporting farmers in preventing large-scale yield loss

## Conclusion

This project demonstrates how deep learning can be applied in agriculture by detecting Late Blight disease in potato leaves. With improvements and larger datasets, such models can be deployed in real-world farming solutions.
