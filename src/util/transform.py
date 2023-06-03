import os
import cv2 as cv
from cv2 import Mat
import numpy as np


def load_img(img_path: str, path:str):
    file_path = os.path.join(path, img_path)
    return cv.imread(file_path)

def original_to_canny(img: Mat, save_path: str, filename: str):
    canny = cv.Canny(img, 100, 200)
    cv.imwrite(os.path.join(save_path,filename), canny)

def original_to_harris(img: Mat, save_path: str, filename: str):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    harris = harris_corner(img)
    cv.imwrite(os.path.join(save_path,filename), harris)

def harris_corner(img, block_size=2, ksize=3, threshold=0.001):
    """Detect corners and thresholds them (From COMP 4102 Assignment 2)

    Args:
        img (Mat): Input Image for corners to be detected
        block_size (int, optional): Neighborhood size. Defaults to 2.
        ksize (int, optional): Aperture parameter for the Sobel operator. Defaults to 3.
        threshold (float, optional): Threshold value to use for value acceptance or removal. Defaults to 0.01.

    Returns:
        Mat: Output Image after corners detected + Simple Thresholding
    """
    # Empty Image of same size as img
    dst = np.zeros(img.shape, dtype=np.float32)

    cv.cornerMinEigenVal(img, block_size, dst, ksize)

    # Simple Thresholding using THRESH_TOZERO
    # Values are difficult to see due to TOZERO thresholding rather than BINARY
    ret, thresh = cv.threshold(dst, threshold, 255, cv.THRESH_BINARY)

    return thresh
    

def original_to_hough_circle(img: Mat, save_path: str, filename: str):
    cimg = img.copy()
    blurred = cv.medianBlur(img,5) # Play with this value
    img = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    circles = cv.HoughCircles(img,cv.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=0,maxRadius=0) # Max Radius of 50 removes lots of false circles
    
    if(circles is not None):
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            # draw the outer circle
            cv.circle(cimg,(i[0],i[1]),i[2],(0,0,255),1)
    cv.imwrite(os.path.join(save_path,filename), cimg)
    
def original_to_color_quantization(img: Mat, save_path: str, filename: str):
    # Using K-Means Clustering

    # Blur the image
    img = cv.GaussianBlur(img, (3, 3), 0)
    Z = img.reshape((-1,3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 5
    ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    cv.imwrite(os.path.join(save_path,filename), res2)

def main():
    for modality in ['flair', 'segmentation', 't1']:
        for tumor in ['tumoral', 'healthy']:
            path = os.path.join('data', 'set1', 'kMeans', modality, tumor)
            for img_path in os.listdir(path):
                img = load_img(img_path, path)
                original_to_color_quantization(img, path, img_path)

main()