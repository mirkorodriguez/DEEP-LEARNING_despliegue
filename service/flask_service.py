# Developed by Mirko J. Rodríguez mirko.rodriguezm@gmail.com

# ----------------------------
# REST service under port 9000
# ----------------------------

#Import Flask
from flask import Flask, request, jsonify
from flask_cors import CORS

#Import Tensorflow image
import keras
from keras.preprocessing import image

#Import libraries
import os
import numpy as np
from werkzeug.utils import secure_filename
from model_loader import loadModelH5

UPLOAD_FOLDER = 'uploads/images'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

port = int(os.getenv('PORT', 5000))
print ("Port recognized: ", port)

#Initialize the application service
app = Flask(__name__)
CORS(app)
global loaded_model
loaded_model = loadModelH5()

# Functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#Define a default route
@app.route('/')
def main_page():
	return 'REST service is active via Flask!'

# Model route
@app.route('/model/predict/',methods=['POST'])
def predict():
    data = {"success": False}
    if request.method == "POST":
        # check if the post request has the file part
        if 'file' not in request.files:
            print('No file part')
        file = request.files['file']
        # if user does not select file, browser also submit a empty part without filename
        if file.filename == '':
            print('No selected file')
        if file and allowed_file(file.filename):
            print("\nFilename received:",file.filename)
            filename = secure_filename(file.filename)
            tmpfile = ''.join([UPLOAD_FOLDER ,'/',filename])
            file.save(tmpfile)
            print("\nFilename stored:",tmpfile)

            #loading image
            image_to_predict = image.load_img(tmpfile, target_size=(224, 224))
            test_image = image.img_to_array(image_to_predict)
            test_image = np.expand_dims(test_image, axis = 0)
            test_image = test_image.astype('float32')
            test_image /= 255.0

            predictions = loaded_model.predict(test_image)[0]
            index = np.argmax(predictions)
            CLASSES = ['Daisy', 'Dandelion', 'Rosa', 'Girasol', 'Tulipán']
            ClassPred = CLASSES[index]
            ClassProb = predictions[index]

            print("Classes:", CLASSES)
            print("Predictions",predictions)
            print("Predicción Index:", index)
            print("Predicción Label:", ClassPred)
            print("Predicción Prob: {:.2%}".format(ClassProb))

            #Results as Json
            data["predictions"] = []
            r = {"label": ClassPred, "score": float(ClassProb)}
            data["predictions"].append(r)

            #Success
            data["success"] = True

    return jsonify(data)

# Run de application
app.run(host='0.0.0.0',port=port, threaded=False)
