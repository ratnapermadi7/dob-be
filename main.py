import io
import os
import time

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from PIL import Image

# Constants and configurations
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "app", "model", "saved_model")
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

# Class labels
CLASS_LABELS = {
    0: '1st degree burn',
    1: '2nd degree burn',
    2: '3rd degree burn',
}

CLASS_ID = {
    0: 1,
    1: 2,
    2: 3
}

# Global variables
model = None
model_type = None
input_shape = (224, 224)

# Import the deprecated functions but suppress warnings
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from tensorflow.keras.applications.vgg16 import preprocess_input
    from tensorflow.keras.preprocessing import image as keras_image

def preprocess_image(img):
    """Preprocess image using the same approach as original code"""
    # Import the deprecated functions but ignore warnings
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        from tensorflow.keras.applications.vgg16 import preprocess_input
        from tensorflow.keras.preprocessing import image as keras_image
    
    # Resize image to expected input size
    img = img.resize(input_shape)
    
    # Convert to numpy array using keras function (for consistency with original code)
    img_array = keras_image.img_to_array(img)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    # Preprocess using VGG16 preprocess_input function
    img_array = preprocess_input(img_array)
    
    return img_array

def load_model():
    """Load the TensorFlow model with a simplified, consistent approach"""
    global model, model_type
    
    print(f"Loading model from: {MODEL_PATH}")
    
    try:
        # First try: load as a SavedModel with signatures
        model = tf.saved_model.load(MODEL_PATH)
        model_type = "saved_model"
        
        # Check for signatures and use serving_default if available
        if hasattr(model, 'signatures') and "serving_default" in model.signatures:
            print("Using serving_default signature")
            model = model.signatures["serving_default"]
            model_type = "signature"
            
            # Print input specs for debugging
            if hasattr(model, 'structured_input_signature'):
                print(f"Input specs: {model.structured_input_signature}")
            
        print(f"Model loaded successfully as {model_type}")
        return True
        
    except Exception as e:
        print(f"Error loading SavedModel: {e}")
        
        try:
            # Second try: load as Keras model
            model = tf.keras.models.load_model(MODEL_PATH)
            model_type = "keras"
            
            # Print model summary for debugging
            model.summary()
            
            print("Model loaded successfully as Keras model")
            return True
            
        except Exception as e2:
            print(f"Error loading Keras model: {e2}")
            model = None
            model_type = None
            return False

def predict_with_retry(img_array, model_obj, max_retries=MAX_RETRIES):
    """Make prediction with retry mechanism"""
    for attempt in range(max_retries):
        try:
            print(f"Prediction attempt {attempt+1}/{max_retries}")
            
            # Handle different model types
            if model_type == 'keras':
                return model_obj.predict(img_array)
                
            elif model_type == 'signature':
                # Convert numpy array to tensor
                tensor_input = tf.constant(img_array, dtype=tf.float32)
                result = model_obj(tensor_input)
                
                # Extract prediction from result dictionary
                if isinstance(result, dict):
                    # Find the output tensor
                    if len(result) == 1:
                        key = list(result.keys())[0]
                    else:
                        # Try common output names
                        possible_keys = ['dense', 'predictions', 'output', 'logits', 'probs', 'probabilities']
                        key = next((k for k in possible_keys if k in result), list(result.keys())[0])
                    
                    return result[key].numpy()
                
                # Handle direct tensor output
                if hasattr(result, 'numpy'):
                    return result.numpy()
                
                return result
                
            else:  # Direct SavedModel
                # First try with direct call
                try:
                    result = model_obj(img_array)
                except Exception as e:
                    print(f"Direct call failed: {e}, trying with tensor conversion")
                    # Try with tensor conversion
                    tensor_input = tf.constant(img_array, dtype=tf.float32)
                    result = model_obj(tensor_input)
                
                # Handle different return types
                if isinstance(result, dict):
                    key = list(result.keys())[0]
                    result = result[key]
                
                if hasattr(result, 'numpy'):
                    return result.numpy()
                    
                return result
                
        except Exception as e:
            print(f"Prediction attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                print("All prediction attempts failed")
                raise e

def predict_image(img_bytes, model_obj):
    """Process an image and return predictions"""
    try:
        # Convert bytes to image
        img = Image.open(io.BytesIO(img_bytes))
        
        # Preprocess image
        img_array = preprocess_image(img)
        
        print(f"Model type: {model_type}")
        print(f"Input shape: {img_array.shape}")
        
        # Make prediction with retry
        predictions = predict_with_retry(img_array, model_obj)
        
        print(f"Prediction shape: {predictions.shape if hasattr(predictions, 'shape') else type(predictions)}")
        
        # Ensure predictions is a numpy array
        if not isinstance(predictions, np.ndarray):
            predictions = np.array(predictions)
        
        # Get top prediction
        if len(predictions.shape) > 1:
            top_index = np.argmax(predictions[0])
            confidence = float(predictions[0][top_index]) * 100
        else:
            top_index = np.argmax(predictions)
            confidence = float(predictions[top_index]) * 100
        
        print(f"Top index: {top_index}, Confidence: {confidence:.2f}%")
        
        # Format result as an array with a single JSON object
        result = [{
            "class_id": CLASS_ID.get(top_index),
            "confidence": confidence,
            "label": CLASS_LABELS.get(top_index, f"Class {top_index}")
        }]
        
        return result
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error during prediction: {str(e)}")
        return None

# Flask application
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return jsonify({"message": "Image prediction API. Send images via POST request to /predict"})

@app.route('/health', methods=['GET'])
def health():
    global model
    
    if model is None:
        if not load_model():
            return jsonify({
                "status": "error",
                "message": "Model could not be loaded"
            }), 500
    
    return jsonify({
        "status": "healthy",
        "model_type": model_type
    })

@app.route('/predict', methods=['POST'])
def predict():
    global model
    
    # Load model if not already loaded
    if model is None:
        if not load_model():
            return jsonify({
                "error": "Model is not loaded. Please check server logs."
            }), 500
    
    # Check if file exists in request
    if 'file' not in request.files:
        return jsonify({
            "error": "No file found in request. Please send an image file."
        }), 400
    
    try:
        file_bytes = request.files['file'].read()
        if not file_bytes:
            return jsonify({"error": "Empty file received"}), 400
        
        # Print debug info about the image
        try:
            img = Image.open(io.BytesIO(file_bytes))
            print(f"Received image: format={img.format}, size={img.size}, mode={img.mode}")
        except Exception as img_err:
            print(f"Could not analyze image: {img_err}")
        
        # Make prediction
        result = predict_image(file_bytes, model)
        
        if result is None:
            return jsonify({"error": "Failed to process the image. Check server logs for details."}), 500
        
        # Return the result as an array
        return jsonify(result)
        
    except Exception as e:
        import traceback
        trace = traceback.format_exc()
        print(f"Exception in prediction endpoint: {str(e)}\n{trace}")
        return jsonify({"error": f"Error processing request: {str(e)}"}), 500

if __name__ == '__main__':
    # Try to load the model at startup
    if not load_model():
        print("WARNING: Model could not be loaded at startup.")
    
    # Get port from environment variables with fallback to 5000
    port = int(os.environ.get('PORT', 5000))
    
    print(f"Starting Flask App on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)