from flask import Flask, render_template, request, jsonify
import base64
from PIL import Image
import tensorflow
import numpy as np
import cv2 

def resize(image):
    rs = np.zeros((500,500,3),dtype=np.uint8)
    height = image.shape[0]
    width = image.shape[1]
    rs[:height,:width] = image
    new_array = np.expand_dims(rs, axis=0)
    return new_array

def preprocess_image(image_data):
    # Decode image from bytes and convert it to a NumPy array
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    # Perform any necessary preprocessing (e.g., resizing, normalization)
    image = resize(image)
    return image

app = Flask(__name__)

# # load model
# model = joblib.load('model/cat_dog.pkl')

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return 'No file part', 400

    file = request.files['image']
    
    if file.filename == '':
        return 'No selected file', 400
    
    new_model = tensorflow.keras.models.load_model('model/my_model')
    image_data = file.read()
    rs_image = preprocess_image(image_data)
    pred = new_model.predict(rs_image).argmax()
    
    prediction = "Dogs" if pred == 1 else "Cats"
    image_data = base64.b64encode(image_data).decode("utf-8")
    return render_template("result.html", prediction=prediction, img = image_data)
    

if __name__ == '__main__':
    app.run(debug=True)
    