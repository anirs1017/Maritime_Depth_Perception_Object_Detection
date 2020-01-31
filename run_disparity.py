#######################################Imports###################################################
from os.path import abspath, dirname, join, isfile 
import sys
import os
import numpy as np
import cv2
import math
from disparity import compute_conventional_disparity
#################################################################################################

#################################################################################################


#######################################Function Definitions######################################
    
def cuda_compute_disparity(image_right, image_left,
                           window_size, foreground_right,
                           foreground_left,
                           block_shape=(512, 1, 1),
                           grid_shape=(1, 1, 1)):
    assert image_left.shape == image_right.shape
    if image_left.dtype == np.uint8:
        image_left = image_left.astype(np.float32)
        image_right = image_right.astype(np.float32)
    import pycuda.autoinit
    import pycuda.driver as drv
    from pycuda.compiler import SourceModule
    cuda_filename = 'compute_disparity.cu'
    cuda_kernel_source = open(cuda_filename, 'r').read()
    cuda_module = SourceModule(cuda_kernel_source)
    compute_disparity = cuda_module.get_function('computeDisparity')

    img_height = image_left.shape[0]
    img_width = image_left.shape[1]

    calculated_disparity = np.zeros(shape=(img_height, img_width), dtype=np.float32)
    compute_disparity(
        drv.In(image_left),
        drv.In(image_right),
        np.int32(window_size),
        np.int32(img_height),
        np.int32(img_width),
        drv.In(foreground_right),
        drv.In(foreground_left),
        drv.Out(calculated_disparity),
        block=block_shape,
        grid=grid_shape
    )
    return calculated_disparity

def compute_background_mask(left_image, right_image):
    from cv2.bgsegm import createBackgroundSubtractorMOG
    from cv2 import dilate, erode, getStructuringElement
    bgSub = createBackgroundSubtractorMOG()
    bgSub.apply(left_image)
    bg_mask = bgSub.apply(right_image)
    kernel = getStructuringElement(cv2.MORPH_RECT, (30, 30))
    dilated = dilate(bg_mask, kernel)
    return dilated

def calculate_depth_kernel(CAMERA_FOCAL_LENGTH_PIX,DIST_BW_CAMERAS_M, x1,y1,x2,y2, disparity,ksize):
    dx = x2-x1
    dy = y2-y1
    x = math.floor(x1+dx/2)
    y = math.floor(y1+dy/2)
    
    ksize_dx = int(dx/4)
    ksize_dy = int(dy/4)
    
    if ksize_dx < ksize: 
        ksize_dx = ksize
        
    if ksize_dy < ksize:
        ksize_dy = ksize
    
    vals = []

    for r in range(y-ksize_dy,y+ksize_dy):
        for c in range(x-ksize_dx,x+ksize_dx):
            if y<len(disparity) and x<len(disparity[0]) and disparity[r][c]!=0:
                vals.append(disparity[r][c]) 
    
    dist = np.mean(vals)
    if dist == 0 or len(vals)==0:
        return "Far"
    else:
        z = CAMERA_FOCAL_LENGTH_PIX*DIST_BW_CAMERAS_M/dist
        fdist = math.sqrt(x**2+y**2+z**2)
        dist = fdist
    return dist

def calculate_depth(CAMERA_FOCAL_LENGTH_PIX,DIST_BW_CAMERAS_M,dist):
    if dist == 0:
        return "Far"
    return CAMERA_FOCAL_LENGTH_PIX*DIST_BW_CAMERAS_M/dist
#################################################################################################

###########################################Main##################################################

def main():
    
    output_location = 'output_day.avi'
    cap1 = cv2.VideoCapture('vid1.mp4')
    cap2 = cv2.VideoCapture('vid2.mp4')
    cuda_flag = 0
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(cap1.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_location, fourcc,fps , (width,height))
    
    count = 0
    CAMERA_FOCAL_LENGTH_MM = 3.6
    DIST_BW_CAMERAS_M = 12.8016
    CAMERA_SENSOR_WIDTH_MM = 4.54
    
    CAMERA_FOCAL_LENGTH_PIX = (CAMERA_FOCAL_LENGTH_MM*width)/CAMERA_SENSOR_WIDTH_MM
    while(True):
        if count == 1:
            break
        count = count + 1
        f1,left_img = cap1.read()
        f2, right_img = cap2.read()
        if(cuda_flag == 1):
            bg_mask = compute_background_mask(left_img, right_img)
            disparity_img = cuda_compute_disparity(
                    image_left=left_img,
                    image_right=right_img,
                    foreground_left=np.ones(shape=(left_img.shape[0:1]),
                                            dtype=np.uint8),
                    foreground_right=np.ones(shape=(right_img.shape[0:1]),
                                             dtype=np.uint8),
                    window_size= 20,
                    block_shape=(512, 1, 1),
                    grid_shape=(math.ceil(left_img.shape[0]* left_img.shape[1]/512), 1, 1)
                )
        else:
            disparity_img = compute_conventional_disparity(left_img, right_img)
        r1=1177
        c1 = 328
        r2 = 1214
        c2 = 342
        d1 = calculate_depth_kernel(CAMERA_FOCAL_LENGTH_PIX,DIST_BW_CAMERAS_M, r1,c1,r2,c2,3, disparity_img)
        colored_disparity = cv2.cvtColor(disparity_img*5,cv2.COLOR_GRAY2RGB).astype(np.uint8)
        out.write(cv2.flip(colored_disparity,180))   
        if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
            break
    
    cap1.release()
    cap2.release()
    out.release()

#################################################################################################
if __name__=='__main__':
    main()  