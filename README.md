# Iris
# Iris Flower Recognition using Support Vector Classification (SVC)

## Overview

This project focuses on building a machine learning model using Support Vector Classification (SVC) to recognize different species of iris flowers based on their features. The dataset used for training and evaluation is the famous Iris dataset.

## Dataset

The Iris dataset is a well-known dataset in machine learning and consists of measurements for 150 iris flowers from three different species: setosa, versicolor, and virginica. The features include sepal length, sepal width, petal length, and petal width.

## Support Vector Classification (SVC)

Support Vector Classification is a supervised learning algorithm that analyzes data for classification tasks. It works by finding the hyperplane that best separates different classes in the feature space while maximizing the margin.

## Model Training

The Iris dataset was split into training and testing sets. The SVC model was trained on the training set to learn the patterns and relationships between the features and the iris species.

```python
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load and split the dataset
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Create and train the Support Vector Classification model
model = SVC(kernel='linear', C=1.0)
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

