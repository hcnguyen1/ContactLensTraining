#-------------------------------------------------------------------------
# AUTHOR: Chi Hao Nguyen
# FILENAME: naive_bayes.py
# SPECIFICATION: Classify the weather data with confidence score of at least 75%
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1 Hour
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM
 
#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import pandas as pd

dbTraining = []
dbTest = []

#Reading the training data using Pandas
df = pd.read_csv('weather_training.csv')
for _, row in df.iterrows():
    dbTraining.append(row.tolist())

#Transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
X = []
for i in dbTraining:
    if i[1].lower() == 'sunny':
        outlook = 1
    elif i[1].lower() == 'overcast':
        outlook = 2
    else:
        outlook = 3
    temperature = 1 if i[2].lower() == 'cool' else 2 if i[2].lower() == 'mild' else 3
    humidity = 1 if i[3].lower() == 'high' else 2
    wind = 1 if i[4].lower() == 'weak' else 2
    X.append([outlook, temperature, humidity, wind])
#Transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
Y = []
for i in dbTraining:
    Y.append(1 if i[5].lower() == 'yes' else 2)


#Fitting the naive bayes to the data using smoothing
#--> add your Python code here
clf = GaussianNB()
clf.fit(X, Y)

#Reading the test data using Pandas
df = pd.read_csv('weather_test.csv')
for _, row in df.iterrows():
    dbTest.append(row.tolist())

#Printing the header os the solution
#--> add your Python code here
#format row column: Day Outlook Temperature Humidity Wind PlayTennis Confidence 
print("Day Outlook Temperature Humidity Wind PlayTennis Confidence")

#Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here
for i in dbTest:
    if i[1].lower() == 'sunny':
        outlook = 1
    elif i[1].lower() == 'overcast':
        outlook = 2
    else:
        outlook = 3
    temperature = 1 if i[2].lower() == 'cool' else 2 if i[2].lower() == 'mild' else 3
    humidity = 1 if i[3].lower() == 'high' else 2
    wind = 1 if i[4].lower() == 'weak' else 2
    testSample = [outlook, temperature, humidity, wind]
    proba = clf.predict_proba([testSample])[0]
    class_predicted = clf.predict([testSample])[0]
    confidence = max(proba)
    label = 'Yes' if class_predicted == 1 else 'No'
    if confidence >= 0.75:
        print(f"{i[0]} {i[1]} {i[2]} {i[3]} {i[4]} {label} {confidence:.2f}")