import argparse
import requests
import cv2
from PIL import Image
import numpy as np
import jsonpickle

addr = 'http://localhost:5000'

def get_response_json(image_path):
    img = Image.open(image_path)
    img = np.asarray(img)

    # encode image as jpeg
    _, img_encoded = cv2.imencode('.jpg', img)

    # prepare headers for http request
    content_type = 'image/jpeg'
    headers = {'content-type': content_type}

    url = addr + '/api/json'

    # send http request with image and receive response
    response = requests.post(url, data=img_encoded.tobytes(), headers=headers)
    data = jsonpickle.decode(response.text)

    # decode response
    print(data['message'])
    print(data['annotation'])


def get_response_img(image_path, output_path):
    img = Image.open(image_path)
    img = np.asarray(img)

    # encode image as jpeg
    _, img_encoded = cv2.imencode('.jpg', img)

    # prepare headers for http request
    content_type = 'image/jpeg'
    headers = {'content-type': content_type}

    url = addr + '/api/img'

    # send http request with image and receive response
    response = requests.post(url, data=img_encoded.tobytes(), headers=headers)

    # decode image
    data = jsonpickle.decode(response.text)
    img = (data["image"])
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    im = Image.fromarray(img.astype('uint8'), 'RGB')

    # decode response
    print(data['message'])

    # save to file
    im.save(output_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', default = './images/demo.jpg')
    parser.add_argument('--output_path', default = 'result.jpg')
    parser.add_argument('--mode', default = 'img', help='img or json')
    args = parser.parse_args()

    if args.mode == 'img':
        get_response_img(args.image_path, args.output_path)
    else:
        get_response_json(args.image_path)


if __name__ == '__main__':
    main()
