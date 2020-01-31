import numpy as np
from sklearn.preprocessing import normalize
import cv2

def compute_conventional_disparity(imgL, imgR):
    imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    
    # SGBM Parameters -----------------
    window_size = 5
    '''
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=160,             # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=5,
        P1=8 * 3 * window_size ** 2,    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    '''
    '''
    left_matcher = cv2.StereoSGBM_create(
        numDisparities=96,           # max_disp has to be dividable by 16 f. E. HH 192, 256
        disp12MaxDiff=20
    )
    '''
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=-64,
        numDisparities=192,             # max_disp has to be dividable by 16 f. E. HH 192, 256
        P1=8 * 3 * window_size ** 2,    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=10,
        uniquenessRatio=1,
        speckleWindowSize=150,
        speckleRange=2,
        preFilterCap=4,
        mode=cv2.STEREO_SGBM_MODE_HH
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
     
    # FILTER Parameters
    lmbda = 80000
    sigma = 1.2
    visual_multiplier = 1.0
     
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)
     
    displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!
     
    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)
    return filteredImg

def compute_conventional_disparity2(imgL, imgR):   
    imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    cv2
    stereo = cv2.StereoSGBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(imgL,imgR)
    return disparity
