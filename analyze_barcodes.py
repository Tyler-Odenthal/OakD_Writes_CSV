import cv2
import os, glob
import numpy as np
from statistics import mean
from roboflow import Roboflow
from pylibdmtx.pylibdmtx import decode
from PIL import Image

rf = Roboflow(api_key="zByfy0GB9wFAa7p8DVFo")
project = rf.workspace().project("warren-friction-weld-wire")
model = project.version(3).model

font = cv2.FONT_HERSHEY_SIMPLEX
font_color = (255, 255, 255)
font_thickness = 2
font_scale = 1

box_color = (255, 255, 255)
box_thickness = 2
box_scale = 1

distance_color = (255, 255, 255)
distance_thickness = 1

cwd = os.getcwd()

# loop through folder of images
file_path = cwd + "/Barcodes/" # folder of images to test - saved to output
prediction_file_path = cwd + "/PredictionBarcodes/" # folder of images to test - saved to output

# file_path = "images/single_image" # single image - saved to single_output
extention = ".JPEG"
globbed_files = sorted(glob.glob(file_path + '*' + extention))
print(globbed_files)

counter = 0

for image_path in globbed_files:

    print(image_path)

    filename = image_path.split("\\")
    filename = filename[-1]
    filename = filename.split(".")
    filename = filename[0]

    image_clean = cv2.imread(image_path)
    blk = np.zeros(image_clean.shape, np.uint8)

    pixel_ratio_array = []

    # infer on a local image
    response_json = model.predict(image_path, confidence=50, overlap=50).json()
    predictions = response_json["predictions"]
    
    saved_image = model.predict(image_path, confidence=50, overlap=50).save(prediction_file_path + filename + "-prediction.jpg")

    for p in predictions:
        
        class_name = p["class"]
        x0 = p['x'] - p['width'] / 2#start_column
        x1 = p['x'] + p['width'] / 2#end_column
        y0 = p['y'] - p['height'] / 2#start row
        y1 = p['y'] + p['height'] / 2#end_row

        if class_name == "QR_CODE":

            croppedArea = image_clean[int(y0):int(y1), int(x0):int(x1)]
            cv2.imwrite(prediction_file_path + filename + "-cropped-QR.jpg", croppedArea)
            QR_image = cv2.imread(prediction_file_path + filename + "-cropped-QR.jpg")

            decoded = decode(cv2.imread(prediction_file_path + filename + "-cropped-QR.jpg"))
            print(decoded)