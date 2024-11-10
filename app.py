from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

student_names = ['Student1 Name1', 'Student2 Name2', 'Student3 Name3']

# Load the model (adjust path as needed)
MODEL_PATH = '/static/models/Run-9.h5'
model = load_model(MODEL_PATH)

# Define image size expected by your model (adjust as necessary)
IMAGE_SIZE = (120, 90)


@app.route("/", methods=["GET"], strict_slashes=False)
def home():
    return render_template('index.html', students=student_names)

@app.route("/classify", methods=["POST"], strict_slashes=False)
def classify_image():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    # Process the uploaded image
    image = request.files["image"]
    image = Image.open(image)
    image = image.resize(IMAGE_SIZE)
    image = img_to_array(image) / 255.0  # Normalize if necessary
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Predict
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)[0]

    # Map prediction to class labels
    class_labels = ['Glasses/Sunglasses', 'Trousers/Jeans', 'Shoes']  # Adjust as needed
    result = class_labels[predicted_class]

    return jsonify({"class": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

