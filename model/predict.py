import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# Load trained model
model = tf.keras.models.load_model("late_blight_model.keras")

# Class labels
labels = {0: "Healthy", 1: "Late Blight"}

# Path to test images
test_folder = "test_images"  # Put your test images here

for img_file in os.listdir(test_folder):
    img_path = os.path.join(test_folder, img_file)

    # Read and preprocess image
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (128, 128))
    img_norm = img_resized / 255.0
    img_expanded = np.expand_dims(img_norm, axis=0)

    # Prediction
    pred = model.predict(img_expanded)
    class_index = int(pred[0][0] > 0.5)  # 0 for Healthy, 1 for Late Blight
    label = labels[class_index]

    # Display image with prediction
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"{img_file} → {label}")
    plt.axis("off")
    plt.show()
