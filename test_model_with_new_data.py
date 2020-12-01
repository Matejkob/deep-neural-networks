import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

min_max_scaling = MinMaxScaler()

# Get test data from cvs
test_data = pd.read_csv("test_data.csv", index_col=0)

# Get data from entire dataset in order to process scaling properly
training_data_df = pd.read_csv("malware_dataset.csv", index_col=0).drop("classification", axis=1)

# Add test data on the bottom of dataset
input_data = training_data_df.append(test_data)
test_data = pd.DataFrame(test_data, columns=training_data_df.columns.tolist())
input_data = min_max_scaling.fit_transform(input_data)

# take only last scaled row as an input
input_data = [input_data[-1].tolist()]

model = load_model('malware_model.h5')

X = input_data
prediction = model.predict(X)
prediction = prediction[0][0]

print("Malware Prediction --> {}".format(prediction))
