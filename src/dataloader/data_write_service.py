import numpy as np
import cv2 as cv

from dataload import getMnistImage,getMnistLabel

def writeImage(fileDir,resulutionTuple):
    imageList = getMnistImage(r"src/datasets/train-images.idx3-ubyte")
    for i, image in enumerate(imageList):
        image_reshape = np.reshape(image,(resulutionTuple[0],resulutionTuple[1]))
        cv.imwrite(fr"{fileDir}/image_{i}.jpg", image_reshape)

def writeLabels(fileDir):
    labelList = getMnistLabel(r"src/datasets/train-labels.idx1-ubyte")
    for i, label in enumerate(labelList):
        open(fr"{fileDir}/{i}_{int(label)}.txt", "a").close()

# writeImage(r"src/datasets/mnist/images",resulutionTuple=(28,28))
# writeLabels(r"src/datasets/mnist/labels")