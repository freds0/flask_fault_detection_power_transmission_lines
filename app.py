from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import base64
import io
import os

from backend.tf_inference import load_custom_model, generate_inference

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

model = load_custom_model('./exported')
app = Flask(__name__)

@app.route('/api/', methods=["POST"])
def main_interface():
    response = request.get_json()
    data_str = response['image']
    point = data_str.find(',')
    base64_str = data_str[point:]  # remove unused part like this: "data:image/jpeg;base64,"
    image = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(image))
    # convert to rgb
    if(img.mode!='RGB'):
        img = img.convert("RGB")
    # convert to numpy array.
    img_arr = np.array(img)
    # do object detection in inference function.
    results = generate_inference(model, img_arr)
    return jsonify(results)

@app.after_request
def add_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    return response


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
