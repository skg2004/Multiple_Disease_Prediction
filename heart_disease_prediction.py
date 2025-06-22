import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle

"""DATA COLLECTION"""

# Load dataset
heart = pd.read_csv("tri_zip/tri/heart.csv")

# Display dataset shape
print("Dataset Shape:", heart.shape)

# Display first 7 rows
print(heart.head(7))

# Check for missing values
print("Missing Values:\n", heart.isnull().sum())

"""DATA ANALYSIS AND DATA VISUALIZATION"""

# Statistical summary
print(heart.describe())

# Count of target variable
print("Target Value Counts:\n", heart['target'].value_counts())

# Define feature matrix (X) and target vector (y)
x = heart.drop(columns='target', axis=1)
y = heart['target']

"""DATA PROCESSING"""

# Standardize the feature values to improve convergence
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

"""SPLITTING TRAIN AND TEST DATA"""

x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y, test_size=0.2, stratify=y, random_state=2
)

print("Data Shapes:", x.shape, x_train.shape, x_test.shape)

"""TRAINING THE MODEL"""

model = LogisticRegression(max_iter=1000, solver='saga')  # Increased max_iter and changed solver
model.fit(x_train, y_train)

"""ACCURACY SCORE"""

# Accuracy on training data
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(y_train, x_train_prediction)
print('Accuracy on training data:', training_data_accuracy)

# Accuracy on test data
x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(y_test, x_test_prediction)
print('Accuracy on testing data:', test_data_accuracy)

# Save the trained model
pickle.dump(model, open("model.pkl", "wb"))
