import numpy as np
import cv2 as cv
import os

def getMNIST(fileDir=fr"src/datasets/unzip_datasets/mnist"):
    dataDict = {}

    imageList = os.listdir(fr"{fileDir}/images")
    print(imageList)
getMNIST()