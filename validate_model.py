import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
training_data_df = pd.read_csv("malware_dataset.csv", index_col=0)
input_data = scaler.fit_transform(training_data_df.drop("classification", axis=1).head(20000))
input_data = [input_data[1623].tolist()]  # <----- Here change test value 1-1000 malware | 1000-2000 benign

model = load_model('malware_model.h5')

X = input_data
prediction = model.predict(X)
prediction = prediction[0][0]

print("Malware Prediction --> {}".format(prediction))
