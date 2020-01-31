from imageai.Detection import VideoObjectDetection
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import run_disparity
from disparity import compute_conventional_disparity,compute_conventional_disparity2
import cv2
import numpy as np
import math
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

execution_path = os.getcwd()
try:
    os.mkdir(execution_path + '/Output')
except(FileExistsError):
    pass
output_path = execution_path + '/Output/'
color_index = {'bus': 'red',
               'giraffe': 'orange',
               'cup': 'yellow',
               'chair': 'green',
               'elephant': 'pink',
               'truck': 'indigo',
               'motorcycle': 'azure',
               'cow': 'magenta',
               'mouse': 'crimson',
               'sports ball': 'raspberry',
               'horse': 'maroon',
               'cat': 'orchid',
               'boat': 'slateblue',
               'parking meter': 'aliceblue',
               'skis': 'deepskyblue',
               'bicycle': 'olivedrab',
               'skateboard': 'palegoldenrod',
               'train': 'cornsilk',
               'bird': 'bisque',
               'bench': 'salmon',
               'bottle': 'brown',
               'car': 'silver',
               'bowl': 'maroon',
               'airplane': 'lavenderblush',
               'umbrella': 'deeppink',
               'bear': 'plum',
               'traffic light': 'mediumblue',
               'snowboard': 'skyblue',
               'dog': 'springgreen',
               'person': 'honeydew',
               'surfboard': 'palegreen',
               'cake': 'sapgreen',
               'book': 'lawngreen',
               'potted plant': 'greenyellow',
               'stop sign': 'beige'}

cap1 = cv2.VideoCapture('data/wheelhouse_bowfar1_20.avi')
cap2 = cv2.VideoCapture('data/wheelhouse_bowfar2_20.avi')

match_method = 0
templ = cv2.imread('template.png', cv2.IMREAD_COLOR );

output_location = output_path + 'distance_video.avi'
output_location_disparity = output_path + 'disparity_video.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fourccdp = cv2.VideoWriter_fourcc(*'XVID')
height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = int(cap1.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(output_location, fourcc,fps , (width,height))
out_disp = cv2.VideoWriter(output_location_disparity, fourccdp,fps , (width,height))

CAMERA_FOCAL_LENGTH_MM = 3.6
DIST_BW_CAMERAS_M = 12.8016
CAMERA_SENSOR_WIDTH_MM = 4.54
CAMERA_FOCAL_LENGTH_PIX = (CAMERA_FOCAL_LENGTH_MM*width)/CAMERA_SENSOR_WIDTH_MM
kernelsize = 10
  
def forFrame(frame_number, output_array, output_count, returned_frame):
    try:
        f1,left_img = cap1.read()
        f2, right_img = cap2.read()
        if f1 and f2:
            disparity = compute_conventional_disparity(left_img, right_img)
            colored_disparity = cv2.cvtColor(disparity*5,cv2.COLOR_GRAY2RGB).astype(np.uint8)
            findisparity = cv2.flip(colored_disparity,180)
            out_disp.write(findisparity)   
        
            result = cv2.matchTemplate(left_img, templ, match_method)
            cv2.normalize( result, result, 0, 1, cv2.NORM_MINMAX, -1 )
            _minVal, _maxVal, minLoc, maxLoc = cv2.minMaxLoc(result, None)
            if (match_method == cv2.TM_SQDIFF or match_method == cv2.TM_SQDIFF_NORMED):
                matchLoc = minLoc
            else:
                matchLoc = maxLoc
        
            refx1 = matchLoc[0] - 120
            refy1 = matchLoc[1] + 110
            refx2 = matchLoc[0] + 150
            refy2 = matchLoc[1] + 180
            cv2.rectangle(returned_frame, (refx1,refy1), (refx2,refy2), (0,0,0), 2, 8, 0 )
            
            d = run_disparity.calculate_depth_kernel(CAMERA_FOCAL_LENGTH_PIX,DIST_BW_CAMERAS_M,
                                                         refx1,refy1,refx2,refy2, disparity,kernelsize)
            if d=='Far':
                cv2.putText(returned_frame, "Distance = "+d, (refx2,refy1), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2)
            else:
                d = d/8
                cv2.putText(returned_frame, "Distance = "+str(np.round(d,2))+' meters', (refx2,refy1), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2)
            
            for index,obj in enumerate(output_array):
                pnts = obj['box_points']
                d = run_disparity.calculate_depth_kernel(CAMERA_FOCAL_LENGTH_PIX,DIST_BW_CAMERAS_M,
                                                         pnts[0],pnts[1],pnts[2],pnts[3], disparity,kernelsize)
                if d=='Far':
                    cv2.putText(returned_frame, "Distance = "+d, (pnts[2],pnts[1]), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2)
                else:
                    d = d/8
                    cv2.putText(returned_frame, "Distance = "+str(np.round(d,2))+' meters', (pnts[2],pnts[1]), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2)
            
            cv2.imshow('Detection',returned_frame)
            cv2.imshow('Disparity',findisparity)
            try:
                out.write(returned_frame.astype(np.uint8))
            except Exception as inst:
                print(inst)
            if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
                cv2.destroyAllWindows()
                cap1.release()
                cap2.release()
    except Exception as e:
        print(e)

detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("yolo.h5")
detector.loadModel(detection_speed='fast')
detector.detectObjectsFromVideo(
        input_file_path=os.path.join(execution_path, "data/wheelhouse_bowfar2_20.avi"), 
        output_file_path=os.path.join(output_path, "boats_detected_out_f") ,  
        frames_per_second=15,
        per_frame_function=forFrame, 
        minimum_percentage_probability=30, 
        return_detected_frame=True, 
        log_progress=True)
out.release()
out_disp.release()
cap1.release()
cap2.release()
cv2.destroyAllWindows()
plt.close()