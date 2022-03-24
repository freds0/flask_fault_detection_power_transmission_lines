from flask import Flask, request, jsonify, Response
from PIL import Image
import numpy as np
import base64
import io
from io import BytesIO
import os
import jsonpickle
import cv2

from backend.tf_inference import load_custom_model, generate_inference, generate_inference_image

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


# route http posts to this method
# https://johnnn.tech/q/how-to-send-image-and-some-text-to-server-flask/
@app.route('/api/img', methods=['POST'])
def get_img():
    r = request
    # convert string of image data to uint8
    nparr = np.frombuffer(r.data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # convert to numpy array.
    img_arr = np.array(img)
    # do object detection in inference function.
    results = generate_inference_image(model, img_arr)
    # encode image as jpeg
    _, img_encoded = cv2.imencode('.jpg', results)

    response = {'message': 'image received', 'image': img_encoded}
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")


# route http posts to this method
@app.route('/api/json', methods=['POST'])
def get_annotation():
    r = request
    # convert string of image data to uint8
    nparr = np.frombuffer(r.data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # convert to numpy array.
    img_arr = np.array(img)
    # do object detection in inference function.
    results = generate_inference(model, img_arr)
    response = {'message': 'image received', 'annotation': results}
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")


@app.after_request
def add_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    return response


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
