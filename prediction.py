import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

min_max_scaling = MinMaxScaler()

good=0
# Get test data from cvs
test_data = pd.read_csv("malware_dataset.csv", index_col=0)
correct_data = test_data[['classification']]
test_data=test_data.drop("classification",axis=1)
# Get data from entire dataset in order to process scaling properly
training_data_df = pd.read_csv("malware_dataset.csv", index_col=0).drop("classification", axis=1)

# Add test data on the bottom of dataset
for a in range(1000):
    input_data=test_data[100*a+1:100*a+2]
    reault=correct_data[100*a+1:100*a+2]

    input_data = training_data_df.append(input_data)
    test_data = pd.DataFrame(test_data, columns=training_data_df.columns.tolist())
    input_data = min_max_scaling.fit_transform(input_data)

        # take only last scaled row as an input
    input_data = [input_data[-1].tolist()]

    model = load_model('malware_model.h5')

    X = input_data
    prediction = model.predict(X)
    prediction = prediction[0][0]
    g=reault["classification"].values
    if g=="malware":
        d=1
    else:
        d=0
    print("test"+reault["classification"])
    if prediction>0.8 and d==1:
        good=good+1
    if prediction<0.2 and d==0:
        good=good+1
    print("Malware Prediction --> {}".format(prediction))
h=good
print("Skuteczność={}%".format(h))
