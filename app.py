from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
model = load_model('skin_cancer_cnn.h5')


def predict_skin_cancer(image_path, model):
    """Preprocess the image and predict skin cancer."""
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    class_label = "Malignant" if prediction > 0.5 else "Benign"

    return class_label


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    filename = None

    if request.method == "POST":
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Predict
            prediction = predict_skin_cancer(filepath, model)

    return render_template("index.html", prediction=prediction, filename=filename)


if __name__ == "__main__":
    app.run(debug=True)
