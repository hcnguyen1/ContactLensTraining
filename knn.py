#-------------------------------------------------------------------------
# AUTHOR: Chi Hao Nguyen
# FILENAME: knn.py
# SPECIFICATION: Compute the error rate for a 1NN classifier
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1 Hour
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

#Reading the data in a csv file using pandas
db = []
df = pd.read_csv('email_classification.csv')
for _, row in df.iterrows():
    db.append(row.tolist())

error_count = 0
total_count = 0
#Loop your data to allow each instance to be your test set
for i in db:

    #Add the training features to the 20D array X removing the instance that will be used for testing in this iteration.
    #For instance, X = [[1, 2, 3, 4, 5, ..., 20]].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here
    X = []
    Y = []
    for j, row in enumerate(db):
        if row != i:
            X.append([float(x) for x in row[:-1]])
            Y.append(1 if row[-1].lower() == 'spam' else 0)

    #Transform the original training classes to numbers and add them to the vector Y.
    #Do not forget to remove the instance that will be used for testing in this iteration.
    #For instance, Y = [1, 2, ,...].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here
    for j in db:
        if j is not i:
            X.append([float(x) for x in j[:-1]])
            Y.append(1 if j[-1].lower() == 'spam' else 0)

    #Store the test sample of this iteration in the vector testSample
    #--> add your Python code here
    testSample = [float(x) for x in i[:-1]]
    true_label = 1.0 if i[-1].lower() == 'spam' else 0.0
    #Fitting the knn to the data using k = 1 and Euclidean distance (L2 norm)
    #--> add your Python code here
    clf = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    clf.fit(X, Y)

    #Use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2, 3, 4, 5, ..., 20]])[0]
    #--> add your Python code here
    class_predicted = clf.predict([testSample])[0]
    #Compare the prediction with the true label of the test instance to start calculating the error rate.
    #--> add your Python code here
    if class_predicted != true_label:
        error_count += 1
    total_count += 1

#Print the error rate
#--> add your Python code here
error_rate = error_count / total_count
print("Error rate:", error_rate)



