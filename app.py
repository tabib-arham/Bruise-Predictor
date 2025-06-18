from flask import Flask, render_template, request
import numpy as np
import os

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = load_model("efficientnetb0_binary_classifier.h5")  # Ensure correct path

# Class mapping (edit if needed)
class_names = {
    0: "Bruised",
    1: "Healthy"
}

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    # Get uploaded file
    imagefile = request.files['imagefile']
    image_path = os.path.join("static", imagefile.filename)

    # Create static dir if not exists
    if not os.path.exists("static"):
        os.makedirs("static")

    try:
        imagefile.save(image_path)
    except FileNotFoundError:
        return "Error: Could not save image. Check folder permissions."

    # Load and preprocess image
    image = load_img(image_path, target_size=(224, 224))  # Change if your model was trained on different size
    image = img_to_array(image) / 255.0  # Rescale to [0, 1]
    image = np.expand_dims(image, axis=0)  # Shape: (1, 224, 224, 3)

    # Predict
    predictions = model.predict(image)

    # Check model output shape
    if predictions.shape[1] == 1:
        # Binary classification (sigmoid)
        prob = float(predictions[0][0])
        predicted_class_index = int(prob >= 0.5)
        confidence = prob if predicted_class_index else 1 - prob
    else:
        # Multiclass classification (softmax)
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_index])

    predicted_class_name = class_names[predicted_class_index]
    classification = f"Class {predicted_class_index}: {predicted_class_name} (Confidence: {confidence:.2f})"

    return render_template('index.html', prediction=classification, image_url=image_path)

if __name__ == '__main__':
    app.run(port=3000, debug=True)
