from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import glob
import cv2
import os

def load_image_data(df, inputPath):
    images = []
    for i in df.index.values:
        print("Proses Data ke ",i)
        basePath = os.path.sep.join([inputPath, "{}_*".format(i + 1)])
        imagePaths = sorted(list(glob.glob(basePath)))
        # print(i)
        inputImages = []
        outputImage = np.zeros((32, 32, 3), dtype="uint8")
        # print(imagePaths)
        for imagePath in imagePaths:
            image = cv2.imread(imagePath)
            image = cv2.resize(image, (32,32))
            inputImages.append(image)

        outputImage[0:32, 0:32] = inputImages[0]
        images.append(outputImage)
    return np.array(images)