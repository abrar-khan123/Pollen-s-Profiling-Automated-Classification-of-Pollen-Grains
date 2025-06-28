from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import uuid

app = Flask(__name__)
model = load_model('model/pollen_cnn_model.h5')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
class_names = ['Pine', 'Oak', 'Maple']  # Replace with actual labels

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def model_predict(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = prediction[0][class_index]
    return f"Predicted class: {class_names[class_index]} with confidence {confidence:.2f}"

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        if 'file' not in request.files:
            prediction = "No file uploaded"
        else:
            file = request.files['file']
            if file.filename == '':
                prediction = "No selected file"
            elif not allowed_file(file.filename):
                prediction = "Invalid file type. Use png/jpg/jpeg"
            else:
                filename = f"{uuid.uuid4().hex}_{file.filename}"
                file_path = os.path.join('static', filename)
                file.save(file_path)
                prediction = model_predict(file_path)
    return render_template('index.html', prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
