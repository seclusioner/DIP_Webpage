from flask import Flask, render_template, request, send_file
import os
from PIL import Image, ImageOps
import cv2
import numpy as np
from io import BytesIO
from DIP import *

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# DIP algorithm settings
def process_image(image, algorithm, params):
    if image.mode != 'RGB':
        image = image.convert('RGB')

    if algorithm == 'grayscale':
        processed_image = ImageOps.grayscale(image)
    elif algorithm == 'binarize':
        threshold = int(params.get('threshold', 128))  # 從 params 中取得閥值，預設為 128
        processed_image = image.convert('L')
        processed_image = processed_image.point(lambda x: 0 if x < threshold else 255, '1')
    elif algorithm == 'canny':
        threshold1 = int(params.get('threshold1', 100))
        threshold2 = int(params.get('threshold2', 200))
        gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        processed_image = cv2.Canny(gray_image, threshold1, threshold2)
        processed_image = Image.fromarray(processed_image)
    elif algorithm == 'HE':
        gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        processed_image = cv2.equalizeHist(gray_image)
        processed_image = Image.fromarray(processed_image)
    elif algorithm == 'Gaussian':
        kernel_size = int(params.get('kernel_size', 5))
        sigma = float(params.get('sigma', 1.0))
        processed_image = GaussianFilter(np.array(image), kernel_size=kernel_size, sigma=sigma)
        processed_image = Image.fromarray(processed_image)
    elif algorithm == 'ErrorDiffusion':
        threshold = int(params.get('threshold', 128))
        processed_image = ErrorDiffusion(np.array(image), threshold=threshold)
        processed_image = Image.fromarray(processed_image)
    else:
        processed_image = image
        
    return processed_image


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            try:
                image = Image.open(file)
            except IOError:
                return 'Invalid file'
            algorithm = request.form['algorithm']
            params = request.form.to_dict()
            processed_image = process_image(image, algorithm, params)
            img_io = BytesIO()
            processed_image.save(img_io, 'PNG')
            img_io.seek(0)
            original_img_io = BytesIO()
            image.save(original_img_io, 'PNG')
            original_img_io.seek(0)
            return send_file(img_io, mimetype='image/png')
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
