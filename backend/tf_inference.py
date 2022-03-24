import tensorflow as tf
import numpy as np
from PIL import Image
from object_detection.utils import ops as utils_ops
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util
import pathlib
import os

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

label_map_path = os.path.join(os.path.dirname(__file__), 'label_map.pbtxt')


def load_model(model_name):
    '''
    Models: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
    '''
    base_url = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(
        fname=model_name,
        origin=base_url + model_file,
        untar=True)
    model_dir = pathlib.Path(model_dir)/"saved_model"
    model = tf.saved_model.load(str(model_dir))
    return model


def load_custom_model(model_name):
    model_file = model_name
    model_dir = pathlib.Path(model_file)/"saved_model"
    model = tf.saved_model.load(str(model_dir))
    return model


def read_label_map(label_map_path):
    '''
    https://stackoverflow.com/questions/55218726/how-to-open-pbtxt-file
    '''
    item_id = None
    item_name = None
    items = {}

    with open(label_map_path, "r") as file:
        for line in file:
            line.replace(" ", "")
            if line == "item{":
                pass
            elif line == "}":
                pass
            elif "id" in line:
                item_id = int(line.split(":", 1)[1].strip())
            elif "name" in line:
                item_name = line.split(":", 1)[1].replace("'", "").strip()

            if item_id is not None and item_name is not None:
                items[item_id] = item_name
                item_id = None
                item_name = None

    return items


def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                           tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict


def generate_inference(model, image_np, conf_thresh=0.5):
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    # Get image dimensions
    height, width, _ = image_np.shape
    # Actual detection.
    output_dict = run_inference_for_single_image(model, image_np)
    # List of the strings that is used to add correct label for each box.
    category_index = label_map_util.create_category_index_from_labelmap(label_map_path, use_display_name=True)
    #category_index = read_label_map(label_map_path)

    boxes = np.array(output_dict['detection_boxes'])
    classes = np.array(output_dict['detection_classes'])
    scores = np.array(output_dict['detection_scores'])

    boxes = np.squeeze(boxes)
    classes = np.squeeze(classes)
    scores = np.squeeze(scores)

    results = []
    for index, score in enumerate(scores):
        if score > conf_thresh:
            label = category_index[classes[index]]['name']
            ymin, xmin, ymax, xmax = boxes[index]

            ymin, ymax = ymin * height, ymax * height
            xmin, xmax = xmin * width, xmax * width

            results.append({"name": label,
                            "conf": str(score),
                            "bbox": [int(xmin), int(ymin), int(xmax), int(ymax)]
                            })

    return {"results": results}


def generate_inference_image(model, image_np):

    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    #image_np = np.array(Image.open(image_path))
    # Actual detection.
    output_dict = run_inference_for_single_image(model, image_np)

    boxes = np.array(output_dict['detection_boxes'])
    classes = np.array(output_dict['detection_classes'])
    scores = np.array(output_dict['detection_scores'])

    boxes = np.squeeze(boxes)
    classes = np.squeeze(classes)
    scores = np.squeeze(scores)

    # List of the strings that is used to add correct label for each box.
    category_index = label_map_util.create_category_index_from_labelmap(label_map_path, use_display_name=True)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=2)

    return image_np
