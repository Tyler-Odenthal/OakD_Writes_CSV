from roboflowoak import RoboflowOak
import cv2
import pandas as pd
import os
import csv
import time
import numpy as np

frame_counter = 0

# field names
fields = ['date', 'timestamp', 'photo_name', 'class_name','x_cord', 'y_cord', 'width', 'height'] 

# name of csv file 
csv_filename = "DetectionLog.csv"
prediction_dir = "PredictionImages"
raw_dir = "RawImages"

cwd = os.getcwd()
print(cwd)

try:
    os.mkdir(prediction_dir)
except:
    pass

try:
    os.mkdir(raw_dir)
except:
    pass

# writing to csv file 
# change 'w' to 'a' for append mode.
with open(csv_filename, 'w') as csvfile: 
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile) 

    # writing the fields 
    csvwriter.writerow(fields) 

    if __name__ == '__main__':
        
        # instantiating an object (rf) with the RoboflowOak module
        # API Key: https://docs.roboflow.com/rest-api#obtaining-your-api-key
        rf = RoboflowOak(model="warren-friction-weld-wire", confidence=0.2, overlap=0.5,
        version="2", api_key="API_KEY", rgb=True,
        depth=False, device=None, blocking=True)

        # Running our model and displaying the video output with detections
        while True:

            t0 = time.time()
            curr_date = time.strftime("%D", time.localtime())
            curr_time = time.strftime("%H:%M", time.localtime())

            # The rf.detect() function runs the model inference
            result, frame, raw_frame, depth = rf.detect()
            predictions = result["predictions"]
            
            # timing: for benchmarking purposes
            t = time.time()-t0
            print("FPS ", 1/t)
            
            for p in predictions:
                    
                    jsonObject = p.json()
                    class_name = jsonObject["class"]
                    x_cord = jsonObject["x"]
                    y_cord = jsonObject["y"]
                    width = jsonObject["width"]
                    height = jsonObject["height"]
                    frame_counter += 1
                    
                    print("PREDICTIONS: " + str(class_name))
                    
                    # Filename
                    filename = class_name + "-" + curr_date.replace("/","-") + "-" + curr_time.replace(":","-") + "-" + str(frame_counter) + ".jpg"
                    
                    # Using cv2.imwrite() method
                    # Saving the image
                    cv2.imwrite(cwd + "/PredictionImages/"+ filename, frame)
                    cv2.imwrite(cwd + "/RawImages/"+ filename, raw_frame)

                    # Dictionary
                    dict = {"date":str(curr_date), "timestamp":str(curr_time), "photo_name":str(filename), "class_name":str(class_name), "x_cord":str(x_cord),"y_cord":str(y_cord), "width":str(width), "height":str(height)}

                    # creating a csv writer object 
                    dict_object = csv.DictWriter(csvfile, fieldnames=fields) 

                    # write dict to CSV
                    dict_object.writerow(dict)

            # displaying the video feed as successive frames
            cv2.imshow("frame", frame)
        
            # how to close the OAK inference window / stop inference: CTRL+q or CTRL+c
            if cv2.waitKey(1) == ord('q'):
                break

df = pd.read_csv(csv_filename)
df.to_csv(csv_filename, index=False)
