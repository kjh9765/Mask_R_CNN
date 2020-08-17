"""
Mask R-CNN
Train on the toy bottle dataset and implement color splash effect.
Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
------------------------------------------------------------
Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:
    # Train a new model starting from pre-trained COCO weights
    python3 bottle.py train --dataset=/home/datascience/Workspace/maskRcnn/Mask_RCNN-master/samples/bottle/dataset --weights=coco
    # Resume training a model that you had trained earlier
    python3 bottle.py train --dataset=/path/to/bottle/dataset --weights=last
    # Train a new model starting from ImageNet weights
    python3 bottle.py train --dataset=/path/to/bottle/dataset --weights=imagenet
    # Apply color splash to an image
    python3 bottle.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>
    # Apply color splash to video using the last weights you trained
    python3 bottle.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import tensorflow as tf
import keras

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
import cv2
from mrcnn.visualize import display_instances
import matplotlib.pyplot as plt

#pip install tensorflow-gpu==1.15.0
#pip install keras==2.2.5
# python final.py train --dataset=/home/ubuntu/Mask_R_CNN/dataset --weight=coco
# tensorboard --logdir=C:/Users/jaehoon/Desktop/medicine_project/logs



# Root directory of the project
ROOT_DIR = os.path.abspath("/home/ubuntu/Mask_R_CNN")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class CustomConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "object"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1
    GPU_COUNT = 4

    # Number of classes (including background)
    NUM_CLASSES = 1 + 166  # Background + toy

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.5


############################################################
#  Dataset
############################################################

class CustomDataset(utils.Dataset):

    def load_custom(self, dataset_dir, subset):
        """Load a subset of the bottle dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("object", 1, "1")
        self.add_class("object", 2, "2")
        self.add_class("object", 3, "3")
        self.add_class("object", 4, "4")
        self.add_class("object", 5, "5")
        self.add_class("object", 6, "9")
        self.add_class("object", 7, "10")
        self.add_class("object", 8, "11")
        self.add_class("object", 9, "12")
        self.add_class("object", 10, "13")

        self.add_class("object", 11, "14")
        self.add_class("object", 12, "15")
        self.add_class("object", 13, "16")
        self.add_class("object", 14, "17")
        self.add_class("object", 15, "18")
        self.add_class("object", 16, "19")
        self.add_class("object", 17, "20")
        self.add_class("object", 18, "21")
        self.add_class("object", 19, "22")
        self.add_class("object", 20, "23")

        self.add_class("object", 21, "24")
        self.add_class("object", 22, "25")
        self.add_class("object", 23, "29")
        self.add_class("object", 24, "30")
        self.add_class("object", 25, "31")
        self.add_class("object", 26, "32")
        self.add_class("object", 27, "33")
        self.add_class("object", 28, "34")
        self.add_class("object", 29, "35")
        self.add_class("object", 30, "36")

        self.add_class("object", 31, "38")
        self.add_class("object", 32, "39")
        self.add_class("object", 33, "40")
        self.add_class("object", 34, "41")
        self.add_class("object", 35, "42")
        self.add_class("object", 36, "43")
        self.add_class("object", 37, "44")
        self.add_class("object", 38, "48")
        self.add_class("object", 39, "49")
        self.add_class("object", 40, "51")

        self.add_class("object", 41, "52")
        self.add_class("object", 42, "53")
        self.add_class("object", 43, "54")
        self.add_class("object", 44, "55")
        self.add_class("object", 45, "56")
        self.add_class("object", 46, "57")
        self.add_class("object", 47, "58")
        self.add_class("object", 48, "59")
        self.add_class("object", 49, "60")
        self.add_class("object", 50, "61")

        self.add_class("object", 51, "62")
        self.add_class("object", 52, "63")
        self.add_class("object", 53, "64")
        self.add_class("object", 54, "65")
        self.add_class("object", 55, "69")
        self.add_class("object", 56, "70")
        self.add_class("object", 57, "71")
        self.add_class("object", 58, "75")
        self.add_class("object", 59, "76")
        self.add_class("object", 60, "77")

        self.add_class("object", 61, "78")
        self.add_class("object", 62, "79")
        self.add_class("object", 63, "80")
        self.add_class("object", 64, "81")
        self.add_class("object", 65, "85")
        self.add_class("object", 66, "86")
        self.add_class("object", 67, "87")
        self.add_class("object", 68, "88")
        self.add_class("object", 69, "89")
        self.add_class("object", 70, "90")

        self.add_class("object", 71, "91")
        self.add_class("object", 72, "92")
        self.add_class("object", 73, "93")
        self.add_class("object", 74, "94")
        self.add_class("object", 75, "95")
        self.add_class("object", 76, "96")
        self.add_class("object", 77, "97")
        self.add_class("object", 78, "98")
        self.add_class("object", 79, "99")
        self.add_class("object", 80, "100")

        self.add_class("object", 81, "101")
        self.add_class("object", 82, "102")
        self.add_class("object", 83, "103")
        self.add_class("object", 84, "104")
        self.add_class("object", 85, "105")
        self.add_class("object", 86, "107")
        self.add_class("object", 87, "108")
        self.add_class("object", 88, "109")
        self.add_class("object", 89, "110")
        self.add_class("object", 90, "111")

        self.add_class("object", 91, "112")
        self.add_class("object", 92, "114")
        self.add_class("object", 93, "115")
        self.add_class("object", 94, "116")
        self.add_class("object", 95, "117")
        self.add_class("object", 96, "118")
        self.add_class("object", 97, "119")
        self.add_class("object", 98, "120")
        self.add_class("object", 99, "121")
        self.add_class("object", 100, "122")

        self.add_class("object", 101, "123")
        self.add_class("object", 102, "125")
        self.add_class("object", 103, "126")
        self.add_class("object", 104, "127")
        self.add_class("object", 105, "128")
        self.add_class("object", 106, "129")
        self.add_class("object", 107, "130")
        self.add_class("object", 108, "131")
        self.add_class("object", 109, "132")
        self.add_class("object", 110, "133")

        self.add_class("object", 111, "134")
        self.add_class("object", 112, "135")
        self.add_class("object", 113, "136")
        self.add_class("object", 114, "137")
        self.add_class("object", 115, "138")
        self.add_class("object", 116, "139")
        self.add_class("object", 117, "140")
        self.add_class("object", 118, "145")
        self.add_class("object", 119, "146")
        self.add_class("object", 120, "147")

        self.add_class("object", 121, "148")
        self.add_class("object", 122, "149")
        self.add_class("object", 123, "150")
        self.add_class("object", 124, "151")
        self.add_class("object", 125, "152")
        self.add_class("object", 126, "153")
        self.add_class("object", 127, "155")
        self.add_class("object", 128, "156")
        self.add_class("object", 129, "157")
        self.add_class("object", 130, "158")

        self.add_class("object", 131, "159")
        self.add_class("object", 132, "160")
        self.add_class("object", 133, "161")
        self.add_class("object", 134, "162")
        self.add_class("object", 135, "163")
        self.add_class("object", 136, "164")
        self.add_class("object", 137, "165")
        self.add_class("object", 138, "166")
        self.add_class("object", 139, "167")
        self.add_class("object", 140, "168")

        self.add_class("object", 141, "169")
        self.add_class("object", 142, "171")
        self.add_class("object", 143, "172")
        self.add_class("object", 144, "173")
        self.add_class("object", 145, "174")
        self.add_class("object", 146, "175")
        self.add_class("object", 147, "176")
        self.add_class("object", 148, "177")
        self.add_class("object", 149, "178")
        self.add_class("object", 150, "179")

        self.add_class("object", 151, "180")
        self.add_class("object", 152, "181")
        self.add_class("object", 153, "182")
        self.add_class("object", 154, "183")
        self.add_class("object", 155, "184")
        self.add_class("object", 156, "185")
        self.add_class("object", 157, "186")
        self.add_class("object", 158, "187")
        self.add_class("object", 159, "188")
        self.add_class("object", 160, "189")

        self.add_class("object", 161, "190")
        self.add_class("object", 162, "195")
        self.add_class("object", 163, "196")
        self.add_class("object", 164, "197")
        self.add_class("object", 165, "198")
        self.add_class("object", 166, "199")

        dataset_dir = CUSTOM_DIR = "/home/ubuntu/Mask_R_CNN/dataset"
        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        annotations1 = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations1.values())  # don't need the dict keys


        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # print(a)
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            #polygons = [r['shape_attributes'] for r in a['regions']]
            polygons = [r['shape_attributes'] for r in a['regions'].values()]
            #objects = [s['region_attributes']['object'] for s in a['regions']]
            objects = [s['region_attributes'] for s in a['regions'].values()]
            print("objects:",objects)
            name_dict ={"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "9": 6, "10": 7, "11": 8, "12": 9, "13": 10, "14": 11,
             "15": 12, "16": 13, "17": 14, "18": 15, "19": 16, "20": 17, "21": 18, "22": 19, "23": 20, "24": 21,
             "25": 22, "29": 23, "30": 24, "31": 25, "32": 26, "33": 27, "34": 28, "35": 29, "36": 30, "38": 31,
             "39": 32, "40": 33, "41": 34, "42": 35, "43": 36, "44": 37, "48": 38, "49": 39, "51": 40, "52": 41,
             "53": 42, "54": 43, "55": 44, "56": 45, "57": 46, "58": 47, "59": 48, "60": 49, "61": 50, "62": 51,
             "63": 52, "64": 53, "65": 54, "69": 55, "70": 56, "71": 57, "75": 58, "76": 59, "77": 60, "78": 61,
             "79": 62, "80": 63, "81": 64, "85": 65, "86": 66, "87": 67, "88": 68, "89": 69, "90": 70, "91": 71,
             "92": 72, "93": 73, "94": 74, "95": 75, "96": 76, "97": 77, "98": 78, "99": 79, "100": 80,
             "101": 81, "102": 82, "103": 83, "104": 84, "105": 85, "107": 86, "108": 87, "109": 88, "110": 89, "111": 90,
             "112": 91, "114": 92, "115": 93, "116": 94, "117": 95, "118": 96, "119": 97, "120": 98, "121": 99,
             "122": 100, "123": 101, "125": 102, "126": 103, "127": 104, "128": 105, "129": 106, "130": 107,
             "131": 108, "132": 109, "133": 110, "134": 111, "135": 112, "136": 113, "137": 114, "138": 115,
             "139": 116, "140": 117, "145": 118, "146": 119, "147": 120, "148": 121, "149": 122, "150": 123,
             "151": 124, "152": 125, "153": 126, "155": 127, "156": 128, "157": 129, "158": 130, "159": 131,
             "160": 132, "161": 133, "162": 134, "163": 135, "164": 136, "165": 137, "166": 138, "167": 139,
             "168": 140, "169": 141, "171": 142, "172": 143, "173": 144, "174": 145, "175": 146, "176": 147,
             "177": 148, "178": 149, "179": 150, "180": 151, "181": 152, "182": 153, "183": 154, "184": 155,
             "185": 156, "186": 157, "187": 158, "188": 159, "189": 160, "190": 161, "195": 162, "196": 163,
             "197": 164, "198": 165, "199": 166}

            #key = tuple(name_dict)
            key = [int(n['object']) for n in objects]
            key2 = key.pop()
            tmp = name_dict[str(key2)]
            num_ids = [tmp]

            #num_ids = [name_dict[a] for a in objects]
            print(num_ids)


            # num_ids = [int(n['Event']) for n in objects]
            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            #print("numids",num_ids)
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "object",  ## for a single class just add the name here
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                num_ids=num_ids)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a bottle dataset image, delegate to parent class.


        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        if info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)
        num_ids = info['num_ids']
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
        	rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])

        	mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        # Map class names to class IDs.
        num_ids = np.array(num_ids, dtype=np.int32)
        return mask, num_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "object":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = CustomDataset()
    dataset_train.load_custom(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CustomDataset()
    dataset_val.load_custom(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")

    MODEL_DIR = '/home/ubuntu/Mask_R_CNN/logs'

    model_inference = modellib.MaskRCNN(mode="inference", config=config, model_dir=MODEL_DIR)

   

    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=10000,
                layers='3+',
                custom_callbacks=[
                                  keras.callbacks.TensorBoard(log_dir=MODEL_DIR,histogram_freq=0, write_graph=True, write_images=False)])




def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]
    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # We're treating all instances as one, so collapse the mask into one layer
    mask = (np.sum(mask, -1, keepdims=True) >= 1)
    # Copy color pixels from the original color image where mask is set
    if mask.shape[0] > 0:
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)

############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect custom class.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/custom/dataset/",
                        help='Directory of the custom dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = CustomConfig()
    else:
        class InferenceConfig(CustomConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 4
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
