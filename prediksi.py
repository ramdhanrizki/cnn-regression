from tensorflow.keras.models import load_model
from utils import dataset
import pandas as pd

df = pd.read_csv("data_film1.csv", sep=";", encoding="iso-8859-1")
df['Pendapatan_int'] = df['Pendapatan'].str.replace(".","")
df["Pendapatan_int"] = df["Pendapatan_int"].astype('int64')

images = dataset.load_image_data(df, "dataset")
images = images / 255.0
maxPendapatan = df["Pendapatan_int"].max()
df["Pendapatan_normalized"] = df["Pendapatan_int"] / maxPendapatan

model = load_model('model_regression')

df["Pendapatan_Prediksi"] = model.predict(images)
df["Hasil_Prediksi"] = df["Pendapatan_Prediksi"] * maxPendapatan
df.to_excel("hasil_prediksi.xlsx")  