import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Base directory for model and prediction
BASE_DIR = os.path.join(os.getcwd(), 'drive', 'MyDrive', 'Outfit Classification')
# Updated model path to .h5 format
MODEL_DIR = os.path.join(BASE_DIR, 'saved_models', 'Model 1', 'Run-1.h5')

# Function to classify a single image
def classify_image(image_path, model_path=MODEL_DIR):
    # Load the trained model
    model = tf.keras.models.load_model(model_path)
    
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(120, 90))  # Resize to match the model's input shape
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Scale pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict the class
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    
    # Define class names
    class_names = ['Glasses/Sunglasses', 'Trousers/Jeans', 'Shoes']
    predicted_label = class_names[predicted_class]

    print(f"Predicted class: {predicted_label}")
    return predicted_label

# Use the function to classify an image
image_path = '/predict/image.jpg'
classify_image(image_path)

