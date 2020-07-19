#%%
from tensorflow.keras.models import load_model
from utils import dataset
import pandas as pd
import glob
import cv2
import math
import xlsxwriter
import numpy as np
import matplotlib.pyplot as plt
#%%
df = pd.read_csv("data_film1.csv", sep=";", encoding="iso-8859-1")
df['Pendapatan_int'] = df['Pendapatan'].str.replace(".","")
df["Pendapatan_int"] = df["Pendapatan_int"].astype('int64')

images, path = dataset.load_image_data(df, "dataset")
images = images / 255.0
maxPendapatan = df["Pendapatan_int"].max()
df["Pendapatan_normalized"] = df["Pendapatan_int"] / maxPendapatan

model = load_model('model_regression')

df["Pendapatan_Prediksi"] = model.predict(images)
df["Hasil_Prediksi"] = df["Pendapatan_Prediksi"] * maxPendapatan
df.to_excel("hasil_prediksi.xlsx")  
#%% Print Images
for i in np.arange(3):
     img = glob.glob(path[i])[0]
     # imgplot = plt.imshow(testImagesX[i], interpolation='nearest')
     plt.imshow(cv2.imread(img))
     plt.suptitle("{} {} ({})".format(i,df['Judul'][i],df['Pendapatan'][i]))
     plt.title("Prediksi : {}".format(dataset.formatangka(math.floor(df['Hasil_Prediksi'][i]))))
     plt.show()

# %%
