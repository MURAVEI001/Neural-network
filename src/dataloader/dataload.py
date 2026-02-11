from pathlib import Path
from PIL import Image
import numpy as np
from natsort import natsorted

def getMnist(dataCount):
    image_Path = Path(r"src/datasets/unzip_datasets/mnist/images")
    label_Path = Path(r"src/datasets/unzip_datasets/mnist/labels")
    imageList = natsorted(image_Path.glob("*.jpg"),key=lambda x: x.name)
    labelList = natsorted(label_Path.glob("*.txt"),key=lambda x: x.name)
    datasetDict = {}
    for i, (image, label) in enumerate(zip(imageList,labelList)):
        datasetDict[f"{i}"] = (np.reshape(np.array(Image.open(image))/255,-1),label.open('r').read())
        if i+1 == dataCount:
            return datasetDict