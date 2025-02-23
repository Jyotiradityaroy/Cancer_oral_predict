import numpy as np
import tensorflow.lite as tflite
from flask import Flask, request, jsonify
from PIL import Image
import io

app = Flask(__name__)

# Load the TFLite model
interpreter = tflite.Interpreter(model_path="new_oral_best_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(image):
    """Preprocess the image to match model input shape (128,128,3)."""
    image = image.resize((128, 128))  # Resize image
    image = np.array(image, dtype=np.float32) / 255.0  # Normalize pixel values
    if image.shape[-1] != 3:
        image = np.stack((image,) * 3, axis=-1)  # Convert grayscale to RGB
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})
    
    file = request.files["file"]
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    processed_image = preprocess_image(image)
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], processed_image)
    interpreter.invoke()
    
    # Get output tensor
    prediction = interpreter.get_tensor(output_details[0]['index'])
    
    label = "Oral Cancer Detected" if prediction[0][0] > 0.5 else "Oral Cancer Not Detected"
    
    return jsonify(label)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))  # Default to 10000, but Render provides a port dynamically
    app.run(host="0.0.0.0", port=port, debug=True)
