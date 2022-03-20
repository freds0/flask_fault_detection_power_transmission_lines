# Flask with Tensorflow 2 for Object Detection Demo

- Frontend and backend are separated, not with common flask template. 

![demo page](images/flask-object-detection.png)
## Server part

Since flask is very simple and wroted by python, we build it with only a few lines of code.

This function receive base64 encoded image from front end page, converted it to PIL Image, then do the object detection step.

```
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
```

## Front end part
In front end page, with the help of  jQuery ajax, we can send the base64 image to backend, wait for the result, then draw the bounding box on the page.

Core code is:
```
// handle image files uploaded by user, send it to server, then draw the result.
function parseFiles(files) {
  const file = files[0];
  const imageType = /image.*/;
  if (file.type.match(imageType)) {
    warning.innerHTML = '';
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onloadend = () => {
      image.src = reader.result;
      // send the img to server
      communicate(reader.result);
    }
  } else {
    setup();
    warning.innerHTML = 'Please drop an image file.';
  }
}
```

## How to run.
Step 1: Open the server with 
```
python app.py

```
The server is setup on port 5000.

Step 2: Open the front end page.

If you want to use python.
```
// python3
python -m http.server
// python2
python -m SimpleHTTPServer

```
If you prefer Node.js
```
npm install serve -g // install serve
serve // this will open a mini web serve
// or http-serve
npm install http-server -g
http-server
```

## Demo
You can click the button, or drag one image to the page, then the result will show bellow.

![页面效果图](images/example.png)

## Run by Docker

### Quick Start：

If you have docker installed, you can quickly test demo:

`docker run -p 5000:5000 -p 8000:8000 -d object_detection`

### Docker Build：

If you want to build your own docker image, you can modify the Dockerfile and execute the command:

`docker build -t object_detection .`

`docker run -p 5000:5000 -p 8000:8000 -d object_detection`


## Client

You can use client.py to access the service. In the clinet.py file it is necessary to configure the ip of the server, setting the variable:
```
addr = 'http://localhost:5000'
```

Then just run the following command:

```
$ python client.py --image_path=IMAGE_FILEPATH --output_path=OUTPUT_FILEPATH --mode=img
```

In this way, the image will be returned with the annotations provided by the model. Running the following command a json file will be returned containing the annotation predicted by the model:

```
$ python client.py --image_path=IMAGE_FILEPATH --output_path=OUTPUT_FILEPATH --mode=json
```


### Demo Results:

![image_test_1](images/example0.jpg)
![image_test_2](images/example1.jpg)
![image_test_3](images/example2.jpg)
![image_test_4](images/example3.jpg)

## References:

- Original Code: [https://github.com/AIZOOTech/flask-object-detection](https://github.com/AIZOOTech/flask-object-detection)
