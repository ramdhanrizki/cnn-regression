from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
import glob
import cv2
import os
from sklearn.model_selection import train_test_split
from utils import dataset
from utils import models
import locale
import math 
from sklearn.metrics import mean_squared_error

df = pd.read_csv("data_film1.csv", sep=";", encoding="iso-8859-1")
df['Pendapatan_int'] = df['Pendapatan'].str.replace(".","")
df["Pendapatan_int"] = df["Pendapatan_int"].astype('int64')

images = dataset.load_image_data(df, "dataset")
images = images / 255.0

split = train_test_split(df, images, test_size=0.25, random_state=42)
(trainAttrX, testAttrX, trainImagesX, testImagesX) = split

maxPendapatan = trainAttrX["Pendapatan_int"].max()
trainY = trainAttrX["Pendapatan_int"] / maxPendapatan
testY = testAttrX["Pendapatan_int"] / maxPendapatan

model = models.create_cnn(32, 32, 3, regress=True)
opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)


print("[INFO] training model...")
model.fit(x=trainImagesX, y=trainY, 
     validation_data=(testImagesX, testY),
     epochs=93, batch_size=8)

print("[INFO] prediksi penghasilan...")
preds = model.predict(testImagesX)

mse = mean_squared_error(testY, preds)
rmse = math.sqrt(mse)

print("Nilai MSE : ", mse)
print("Nilai RMSE : ", rmse)

# Simpan Model 
model.save('model_regression')