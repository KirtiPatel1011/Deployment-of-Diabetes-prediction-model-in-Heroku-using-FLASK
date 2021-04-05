import numpy as np
import pandas as pd
import pickle


#Loading the dataset
diabetes=pd.read_csv("diabetes.csv",header=0)
diabetes.head(5)
diabetes = diabetes.rename(columns={'DiabetesPedigreeFunction':'DPF'})

# Replacing the 0 values from ['Glucose','BloodPressure','SkinThickness','Insulin','BMI'] by NaN
diabetes_copy = diabetes.copy(deep=True)
diabetes_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = diabetes_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NAN)

# Replacing NaN value by mean, median depending upon distribution
diabetes_copy['Glucose'].fillna(diabetes_copy['Glucose'].mean(), inplace=True)
diabetes_copy['BloodPressure'].fillna(diabetes_copy['BloodPressure'].mean(), inplace=True)
diabetes_copy['SkinThickness'].fillna(diabetes_copy['SkinThickness'].median(), inplace=True)
diabetes_copy['Insulin'].fillna(diabetes_copy['Insulin'].median(), inplace=True)
diabetes_copy['BMI'].fillna(diabetes_copy['BMI'].median(), inplace=True)

#model Building
x=diabetes_copy.iloc[:,0:-1]
y=diabetes_copy.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=0)
# Creating Random Forest Model
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=20,random_state=0)
classifier.fit(x_train,y_train)

# Creating a pickle file for the classifier
filename = 'model.pkl'
pickle.dump(classifier, open(filename, 'wb'))


