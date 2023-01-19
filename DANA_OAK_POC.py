from roboflowoak import RoboflowOak
from pylibdmtx.pylibdmtx import decode
import cv2
import pandas as pd
import os
import csv
import time
import numpy as np

# roboflow variables
api_key = "API"
version_number = "3"
model_id = "columbia-carrier-v2"
record_video = False

# frame logic variables
target_timer_cooldown = 60                  # the amount of frames (wait) between each target location - approximately 2 seconds
target_timer = (0-target_timer_cooldown)    # increments every frame
target_timer_max = 150                      # the amount of frames (wait) before failing a target location and rotate to next targets - approximately 5 seconds
target_timer_min = 15                       # the minimum amount of frames (wait) to pass a target location - approximately .5 seconds
current_target_counter = 0                  # used for incrementing target array
total_loops_count = 0                       # increments based on len(target_objects)
TARGETS_ON = False                          # sets True when cooldown is over

# detection logic variables - ADD "Temp" to increase target objects - amount of "Temp" and target_objects[] must match
target_objects = ["Temp","Temp","Temp","Temp"]

# replace each temp with target objects for each section of logging
target_objects[0] = ["Data-Matrix"]
target_objects[1] = ["Speed-Sensor", "Speed-Sensor-Bolt", "Vent-Cap"]
target_objects[2] = ["Clutch_Motor"]
target_objects[3] = ["Bolt", "Bolt", "Bolt"]
current_targets = target_objects[current_target_counter]

# font cords, style and colors
font = cv2.FONT_HERSHEY_COMPLEX_SMALL   # font style
target_timer_org = (10, 30)             # target_timer text x,y coordinates
targets_org = (10, 50)                  # targets text x,y coordinates
passed_org = (10, 70)                   # passed text x,y coordinates
fps_org = (10, 620)                     # fps text x,y coordinates
fontScale = 1                           # font size
target_timer_color = (0, 0, 200)        # target_timer color - red
passed_color = (0, 0, 200)              # passed color - red
color = (255, 255, 255)                 # default color - white
thickness = 1                           # font thickness

class_name = "None"
decoded_serial = ""
x_cord = 0
y_cord = 0
width = 0
height = 0

# used for determining pass / fail
target_passed = False
target_array = []
fps_array = []
pass_array = []

print(target_objects)

# CSV field names - amount of target_objects[] must match above
fields = ['serial_code', 'date', 'timestamp', 'part_count', 'targets', 'check_passed', str(target_objects[0])+'-0', 'photo_path_0', str(target_objects[1])+'-1', 'photo_path_1', str(target_objects[2])+'-2', 'photo_path_2', str(target_objects[3])+'-3', 'photo_path_3']

# name of csv file 
csv_filename = "DetectionLog.csv"
prediction_dir = "PredictionImages"
raw_dir = "RawImages"

# used for writing to file and CSV
object_counter = 0

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

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
if record_video == True:
    out = cv2.VideoWriter('outpy.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 25, (640,640))

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
        rf = RoboflowOak(model=model_id, confidence=0.2, overlap=0.5,
        version=version_number, api_key=api_key, rgb=True,
        depth=False, device=None, blocking=True)

        # Running our model and displaying the video output with detections
        while True:
            
            if target_timer < 0:
                target_timer_color = (0, 0, 200)
            else:
                target_timer_color = (0, 255, 0)

            t0 = time.time()
            curr_date = time.strftime("%D", time.localtime())
            curr_time = time.strftime("%H:%M", time.localtime())

            # The rf.detect() function runs the model inference
            result, frame, raw_frame, depth = rf.detect()
            predictions = result["predictions"]
            
            # timing: for benchmarking purposes
            t = time.time()-t0
            fps_array.append(1/t)
            fps_array[-150:]
            fps_average = sum(fps_array)/len(fps_array)

            target_timer += 1

            # Using cv2.putText() method
            image = cv2.putText(frame, "Timer: " + str(target_timer), target_timer_org, font, fontScale, target_timer_color, thickness, cv2.LINE_AA, False)
            
            # Using cv2.putText() method
            image = cv2.putText(frame, "Current Targets" + "("+ str(total_loops_count) + ")" + ": " + str(current_targets), targets_org, font, fontScale, color, thickness, cv2.LINE_AA, False)

            # Using cv2.putText() method
            image = cv2.putText(frame, "Passed: " + str(target_passed), passed_org, font, fontScale, passed_color, thickness, cv2.LINE_AA, False)

            # Using cv2.putText() method
            image = cv2.putText(frame, "FPS: " + str(fps_average)[:5], fps_org, font, fontScale, color, thickness, cv2.LINE_AA, False)

            if 0 < target_timer >= target_timer_max and TARGETS_ON == True:
                
                # reset variables
                target_timer = (0-target_timer_cooldown)
                current_target_counter += 1
                target_array = []
                passed_color = (0, 0, 200)
                
                if current_target_counter <= (len(target_objects)):
                    # Saving the image
                    filename = "Inspection-Check-" + curr_date.replace("/","-") + "-" + curr_time.replace(":","-") + "-" + str(object_counter) + ".jpg"
                    cv2.imwrite(cwd + "/PredictionImages/"+ filename, frame)
                    cv2.imwrite(cwd + "/RawImages/"+ filename, raw_frame)

                    # array looping counters
                    print("INCREMENTED TARGET ARRAY")
                    current_targets = target_objects[current_target_counter-1]
                    pass_array.append(target_passed)
                else:
                    if False in pass_array:
                        print("TOTAL FAIL")
                        total_pass = False
                    else:
                        total_pass = True
                        print("TOTAL PASS")
                    
                    # Saving the image
                    filename = "Inspection-Check-" + curr_date.replace("/","-") + "-" + curr_time.replace(":","-") + "-" + str(object_counter) + ".jpg"
                    cv2.imwrite(cwd + "/PredictionImages/"+ filename, frame)
                    cv2.imwrite(cwd + "/RawImages/"+ filename, raw_frame)

                    pass_array.append(target_passed)
                    print(pass_array)

                    pass_check0 = pass_array[0]
                    pass_check1 = pass_array[1]
                    pass_check2 = pass_array[2]
                    pass_check3 = pass_array[3]

                    # Dictionary and write dict to CSV
                    dict = {"serial_code":str(decoded_serial), "date":str(curr_date), "timestamp":str(curr_time), "part_count":str(total_loops_count), "targets":str(target_objects), "check_passed":str(total_pass), str(target_objects[0])+'-0':pass_check0,"photo_path_0":str(cwd + "/PredictionImages/"+ filename), str(target_objects[1])+'-1':pass_check1, "photo_path_1":str(cwd + "/PredictionImages/"+ filename), str(target_objects[2])+'-2':pass_check2, "photo_path_2":str(cwd + "/PredictionImages/"+ filename), str(target_objects[3])+'-3':pass_check3, "photo_path_3":str(cwd + "/PredictionImages/"+ filename)}
                    dict_object = csv.DictWriter(csvfile, fieldnames=fields) 
                    dict_object.writerow(dict)
                    dict = {}
                    
                    # array looping counters
                    print("TARGET ARRAY RESET")
                    current_target_counter = 1
                    current_targets = target_objects[current_target_counter-1]
                    total_loops_count += 1
                    pass_array = []
                    decoded_serial = ""
                
                # reset targets pass / fail
                target_passed = False

            if target_timer == 0 and TARGETS_ON == False:
                print("COOLDOWN OVER")
                target_timer = 0
                current_target_counter = 1
                target_timer_color = (0, 255, 0)
                TARGETS_ON = True

            temp_array = []

            for p in predictions:
                    
                    jsonObject = p.json()
                    class_name = jsonObject["class"]
                    width = jsonObject["width"]
                    height = jsonObject["height"]
                    x0 = jsonObject['x'] - jsonObject['width'] / 2 #start_column
                    x1 = jsonObject['x'] + jsonObject['width'] / 2 #end_column
                    y0 = jsonObject['y'] - jsonObject['height'] / 2 #start row
                    y1 = jsonObject['y'] + jsonObject['height'] / 2 #end_row

                    object_counter += 1
                    
                    # print("PREDICTIONS: " + str(class_name))
                    temp_array.append(class_name)

            target_array.append(temp_array)

            target_results = sum(1 if current_targets == x else 0 for x in target_array)

            # print(target_results)

            if target_results >= target_timer_min and target_passed != True:
                
                if class_name == "Data-Matrix":

                    croppedArea = raw_frame[int(y0):int(y1), int(x0):int(x1)]
                    cv2.imwrite(cwd + "/PredictionBarcodes/" + filename + "-cropped-QR.jpg", croppedArea)
                    QR_image = cv2.imread(cwd + "/PredictionBarcodes/" + filename + "-cropped-QR.jpg")

                    decoded_serial = decode(cv2.imread(cwd + "/PredictionBarcodes/" + filename + "-cropped-QR.jpg"))
                    print(decoded_serial)

                passed_color = (0, 255, 0)
                target_passed = True
                
            # displaying the video feed as successive frames
            cv2.imshow("frame", frame)
            
            if record_video == True:
                out.write(raw_frame)
        
            # how to close the OAK inference window / stop inference: CTRL+q or CTRL+c
            if cv2.waitKey(1) == ord('q'):
                break

df = pd.read_csv(csv_filename)
df.to_csv(csv_filename, index=False)
