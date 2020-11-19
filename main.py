import os
from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

import pickle
import tensorflow as tf

from keras.models import load_model
# from keras.backend import set_session

import numpy as np
import matplotlib.pyplot as plt
import cv2


ALLOWED_EXTENSIONS = set(['png','jpg','jpeg','gif'])
em_arr = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

# tf.compat.v1.disable_eager_execution()

# global sess
# sess = tf.compat.v1.Session()

# global graph
# graph = tf.compat.v1.get_default_graph()

# init = tf.compat.v1.global_variables_initializer()
# sess.run(init)

# set_session(sess)
model = load_model('./ml_models/model_2.h5')
# model = pickle.load(open('./ml_models/model.pkl','rb'))


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/', methods = ['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('Missing file part')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash('Please select an image for uploading')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        flash('Upload successful.\nDisplay successful.')

        file_names = []
        file_names.append(filename)

        image = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (48,48))
        image = image / 255
        image = np.expand_dims(image, axis=2)

        zeros_4dim = np.zeros(shape=(1,48,48,1))
        zeros_4dim[0] = image
    
        # with graph.as_default():
        #     set_session(sess)
        predicted_probs = model.predict(zeros_4dim)
        
        plt.cla(); plt.clf(); plt.close();
        plt.bar(em_arr, height=predicted_probs[0])
        plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], f'prediction_{filename}.png'))

        file_names.append(f'prediction_{filename}.png')

        return render_template('upload.html', filenames=file_names)

    else:
        flash('The accepted image types are: png, jpg, jpeg, gif')
        return redirect(request.url)



@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename=f'uploads/{filename}', code = 301))

if __name__ == '__main__':
    app.run(debug=True, threaded = False)