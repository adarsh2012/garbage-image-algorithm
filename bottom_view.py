import cv2
import os, sys, random, math, time
import shutil, glob
from common_func import *


def isGoodImage(imgpath, houghParams, edgeParams):
    name = os.path.split(imgpath)[1]
    img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
    edgeImg = getBottleEdges(img, mask=None, thresh=edgeParams["validThresh"])
    circles = cv2.HoughCircles(edgeImg,
                               cv2.HOUGH_GRADIENT,
                               houghParams["res"],
                               houghParams["minDist"],
                               param1=houghParams["param1"],
                               param2=houghParams["param2"],
                               minRadius=houghParams["minRadius"],
                               maxRadius=houghParams["maxRadius"])
    if circles is None:
        return 0
    else:
        return 1

if __name__ == "__main__":
    hf_param = {"res": 1.5, "minDist": 1000, "param1": 200, "param2": 60, "minRadius": 300, "maxRadius": 550}
    eg_param = {"validThresh": 40}
    print("Hough - {}\nEdge - {}".format(hf_param, eg_param))
    imgLoc = input("Image location: ")
    storeLoc = os.path.join(imgLoc, "BadImages")
    os.mkdir(storeLoc)
    print("Starting....")
    for im in glob.glob(imgLoc + "/**/*.tiff", recursive=True):
        print(im)
        pred = isGoodImage(im, houghParams=hf_param, edgeParams=eg_param)
        if(pred == 0):
            shutil.move(im, storeLoc)
        
