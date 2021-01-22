import cv2
import numpy as np
import os, sys, random, math, time
import glob


def displayImg(img, isHSV=False, title=""):
    if(isHSV):
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    if(len(img.shape) <= 2):
        pyplot.imshow(img, cmap = 'gray', vmin=0, vmax=255)
    else:
        pyplot.imshow(img)
    pyplot.title(title)
    pyplot.show()

def segment_hsv(img, mask):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    #hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
    return cv2.inRange(hsv, mask[0], mask[1])

def getSob(img, size):
    #sobel
    img_sobelx = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=size)
    img_sobely = cv2.Sobel(img,cv2.CV_8U,0,1,ksize=size)
    x_filtered_image = cv2.convertScaleAbs(img_sobelx)
    y_filtered_image = cv2.convertScaleAbs(img_sobely)
    img_sobel = cv2.addWeighted(x_filtered_image, 0.5, y_filtered_image, 0.5, 0)
    return img_sobel, x_filtered_image, y_filtered_image

def getBottleEdges(img, mask=None, thresh = 50, gsize=15, sobsize=5):
    img_blurr = cv2.GaussianBlur(img, (gsize,gsize), cv2.BORDER_DEFAULT)
    img_fin, img_x, img_y = getSob(cv2.cvtColor(img_blurr, cv2.COLOR_RGB2GRAY), sobsize)
    if(mask != None):
        img_fin = cv2.bitwise_and(img_fin, img_fin, mask=segment_hsv(img_blurr, mask))
    if(thresh != None):
        img_fin[img_fin >= thresh] = 255
        img_fin[img_fin < thresh] = 0
    return img_fin


def countFiles(path):
    count = 0
    for im in glob.glob(path+"/**/*.tiff", recursive=True):
        count += 1
    return count

if __name__ == "__main__":
    path = r"D:\Documents\Capstone_Work\Testing_Sample_keras\Bad_images_bottom"
    print(countFiles(path))


