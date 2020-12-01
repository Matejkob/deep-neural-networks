import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import *

min_max_scaling = MinMaxScaler()
training_data_df = pd.read_csv("malware_dataset.csv", index_col=0)

training_data_df["classification"].replace({"malware": 1, "benign": 0}, inplace=True)

# Take only 2000 first rows where we have 1000 malware and 1000 benign
input_data = training_data_df.drop("classification", axis=1)
val_data = training_data_df[['classification']].head(2000)
input_data = input_data.head(2000)

# Proper data scaling based on entire date set (<-- check needed)
input_data = min_max_scaling.fit_transform(input_data)

X = input_data
Y = val_data

# Define the model
model = Sequential()
model.add(Dense(50, input_dim=len(X[0]), activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model
model.fit(
    X,
    Y,
    epochs=30,
    shuffle=True,
    verbose=2
)

test_error_rate = model.evaluate(X, Y, verbose=0)
print("The mean squared error (MSE) for the test data set is: {}".format(test_error_rate))

model.save("malware_model.h5")
print("Model saved to disk.")
