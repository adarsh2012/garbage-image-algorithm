import cv2
import os, sys, random, math, time
import shutil, glob
from common_func import *
import math
import copy

def getTopViewCircles(imgPath, drawSize=2, res=1.5, minRad=350, maxRad=500, param2=50, minDist=1000):
    target_img_RGB = cv2.imread(imgPath, cv2.IMREAD_COLOR)
    target_img_RGB = cv2.cvtColor(target_img_RGB, cv2.COLOR_BGR2RGB)
    target_edgeImg = getBottleEdges(target_img_RGB)
    target_img_withCircles = copy.deepcopy(target_img_RGB)
    circles = cv2.HoughCircles(target_edgeImg,cv2.HOUGH_GRADIENT,res,minDist,param1=150,param2=param2,minRadius=minRad,maxRadius=maxRad)
    if circles is not None:
        for i in circles[0,:]:
            cv2.circle(target_img_withCircles,(i[0],i[1]),int(i[2]),(0,255,0),drawSize)
    return (target_img_RGB, target_img_withCircles, circles, target_edgeImg)
    

def topview_bad_detect(img, circles, ratioThresh=0.3):
    #Circle returns --> (mid1, mid2, radius)
    if(circles is None):
        return False
    detectedCircle = circles[0,:,:][0]
    cornerCases = [
        (detectedCircle[1] + detectedCircle[2], detectedCircle[0]),
        (detectedCircle[1] - detectedCircle[2], detectedCircle[0]),
        (detectedCircle[1], detectedCircle[0] + detectedCircle[2]),
        (detectedCircle[1], detectedCircle[0] - detectedCircle[2])
    ]
    for cases in cornerCases:
        for axis in range(2): #Axis: x,y represented using the index which is 0,1
            ratio = 0
            if(cases[axis] < 0):
                ratio = abs(cases[axis])/detectedCircle[2]
            elif(cases[axis] > img.shape[axis]):
                ratio = abs(cases[axis] - img.shape[axis])/detectedCircle[2]
            if(ratio > ratioThresh):
                return 0
    return 1

def topViewPredictWrap(imgpath, ratio):
    img_rgb, img_with_circle, circles, img_edge = getTopViewCircles(imgpath)
    pred = topview_bad_detect(img_rgb, circles, ratioThresh=ratio)
    return pred

if __name__ == "__main__":
    ratio = 0.3
    print("Ratio thresh: {}".format(ratio))
    procedureWrapper(topViewPredictWrap, {"ratio": ratio}, "top")

