from flask import Flask, render_template, request
import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

# Class labels per model
class_names_dict = {
    "DenseNet121": ['Gir', 'Sahiwal', 'Jersey', 'HF', 'Red Sindhi'],
    "EfficientNet": ['Gir', 'Sahiwal', 'Jersey', 'HF', 'Red Sindhi'],
    "EfficientNetB3": ['Gir', 'Sahiwal', 'Jersey', 'HF', 'Red Sindhi'],
    "MobileNetV2": ['Gir', 'Sahiwal', 'Jersey', 'HF', 'Red Sindhi'],
    "MobileNetV3": ['Gir', 'Sahiwal', 'Jersey', 'HF', 'Red Sindhi'],
    "ConvNext" : ['Gir', 'sahiwal', 'Jersey', 'HF','Red Sindhi']
}

#Load all TFLite models =====
model_paths = {
    "DenseNet121": "models/DenseNet121_cattle_breed_model.tflite",
    "EfficientNet": "models/EfficientNet_cattle_breed_model.tflite",
    "EfficientNetB3": "models/EfficientNetB3_cattle_breed_model.tflite",
    "MobileNetV2": "models/Mobilenetv2_cattle_breed_model.tflite",
    "MobileNetV3": "models/Mobilenetv3_large_cattle_breed.tflite",
    "ConvNext" : "models/convnext_cattle_breed_model.tflite" 
}

interpreters = {}
for name, path in model_paths.items():
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    interpreters[name] = interpreter

# ===== Predict function per model =====
def predict_with_model(interpreter, image, model_name):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Get expected input size
    height, width = input_details[0]['shape'][1], input_details[0]['shape'][2]

    # Resize image for this model
    img_resized = image.resize((width, height))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    # Run the model
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    # Get predicted class index
    pred_index = int(np.argmax(output))
    model_classes = class_names_dict[model_name]

    if pred_index >= len(model_classes):
        pred_index = 0

    confidence = float(np.max(output))
    return model_classes[pred_index], confidence

# ===== Predict across all models =====
def predict(image):
    results = {}
    for name, interpreter in interpreters.items():
        breed, conf = predict_with_model(interpreter, image, name)
        results[name] = {
            "breed": breed,
            "confidence": round(conf * 100, 2)
        }
    return results

# ===== Flask route =====
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        file = request.files['image']
        image = Image.open(file).convert('RGB')
        results = predict(image)
        return render_template('index.html', results=results)
    return render_template('index.html', results=None)

# ===== Run app =====
if __name__ == '__main__':
    app.run(debug=True)
    
    
