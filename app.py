import numpy as np
import os
import uuid
import tensorflow as tf
from flask import Flask, render_template, request, send_file, redirect, url_for, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from datetime import datetime

app = Flask(__name__)

MODEL_PATH = 'finetuned_mobilenetv2_colon_detection_model.h5'

try:
    model = load_model(MODEL_PATH)
    print("✅ Model Loaded Successfully!")
except Exception as e:
    print(f"❌ Error Loading Model: {e}")
    model = None

UPLOAD_FOLDER = 'uploads'
REPORTS_FOLDER = 'reports'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

CLASS_LABELS = ['Normal', 'Ulcerative Colitis', 'Polyps', 'Esophagitis']

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files or 'patient_name' not in request.form or 'patient_age' not in request.form:
            return render_template("predict.html", predict="Incomplete form submission.")

        file = request.files['file']
        patient_name = request.form['patient_name']
        patient_age = request.form['patient_age']

        if file.filename == '':
            return render_template("predict.html", predict="No file selected.")

        if not allowed_file(file.filename):
            return render_template("predict.html", predict="Invalid file format. Please upload a PNG or JPG image.")

        ext = file.filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{uuid.uuid4()}.{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)

        try:
            img = load_img(filepath, target_size=(224, 224))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            if model:
                predictions = model.predict(img_array)
                predicted_class = np.argmax(predictions, axis=1)[0]
                prediction_result = CLASS_LABELS[predicted_class]
            else:
                prediction_result = "Model is not loaded."

        except Exception as e:
            prediction_result = f"⚠️ Error processing image: {e}"

        return render_template("predict.html", predict=prediction_result, patient_name=patient_name, patient_age=patient_age, uploaded_file=unique_filename)

    return render_template("predict.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
