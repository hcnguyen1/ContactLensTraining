#-------------------------------------------------------------------------
# AUTHOR: Chi Hao Nguyen
# FILENAME: decision_tree_2.py
# SPECIFICATION: Implement decision tree classifier to classify data
# FOR: CS 4210- Assignment #2
# TIME SPENT: 4 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn import tree
import pandas as pd

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

#Reading the test data in a csv file using pandas
dbTest = []
df_test = pd.read_csv('contact_lens_test.csv')
for _, row in df_test.iterrows():
    dbTest.append(row.tolist())

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #Reading the training data in a csv file using pandas
    # --> add your Python code here
    dbTraining = []
    df = pd.read_csv(ds)
    for _, row in df.iterrows():
        dbTraining.append(row.tolist())

    #Transform the original categorical training features to numbers and add to the 4D array X.
    #For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    #--> add your Python code here
    for data in dbTraining:
        features = []
        # Age
        if data[0] == 'Young':
            features.append(1)
        elif data[0] == 'Prepresbyopic':
            features.append(2)
        else:  # Presbyopic
            features.append(3)
        # Spectacle Prescription
        if data[1] == 'Myope':
            features.append(1)
        else:  # Hypermetrope
            features.append(2)
        # Astigmatism
        if data[2] == 'No':
            features.append(1)
        else:  # Yes
            features.append(2)
        # Tear Production Rate
        if data[3] == 'Reduced':
            features.append(1)
        else:  # Normal
            features.append(2)
        X.append(features)

    #Transform the original categorical training classes to numbers and add to the vector Y.
    #For instance Yes = 1 and No = 2, Y = [1, 1, 2, 2, ...]
    #--> add your Python code here
    for data in dbTraining:
        if data[4] == 'Yes':
            Y.append(1)
        else:
            Y.append(2)
            
    total_accuracy = 0.0
    #Loop your training and test tasks 10 times here
    for i in range (10):
       # fitting the decision tree to the data using entropy as your impurity measure and maximum depth = 5
       # --> add your Python code here
        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
        clf = clf.fit(X, Y)
        
        num_correct = 0
       #Read the test data and add this data to dbTest
       #--> add your Python code here
        for data in dbTest:
           #Transform the features of the test instances to numbers following the same strategy done during training,
           #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
           #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
           #--> add your Python code here
            features = []
            # Age
            if data[0] == 'Young':
                features.append(1)
            elif data[0] == 'Prepresbyopic':
                features.append(2)
            else:  # Presbyopic
                features.append(3)
            # Spectacle Prescription
            if data[1] == 'Myope':
                features.append(1)
            else:  # Hypermetrope
                features.append(2)
            # Astigmatism
            if data[2] == 'No':
                features.append(1)
            else:  # Yes
                features.append(2)
            # Tear Production Rate
            if data[3] == 'Reduced':
                features.append(1)
            else:  # Normal
                features.append(2)

           #Compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
           #--> add your Python code here
            class_predicted = clf.predict([features])[0]
            if (class_predicted == 1 and data[4] == 'Yes') or (class_predicted == 2 and data[4] == 'No'):
                num_correct += 1
        accuracy = num_correct / len(dbTest)
        total_accuracy += accuracy

    #Find the average of this model during the 10 runs (training and test set)
    #--> add your Python code here
    average_accuracy = total_accuracy / 10.0
    
    #Print the average accuracy of this model during the 10 runs (training and test set).
    #Your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    #--> add your Python code here
    print(f"Final Accuracy {ds}: {average_accuracy}")