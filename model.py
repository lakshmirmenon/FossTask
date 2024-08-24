import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load the dataset from a local file
data = pd.read_csv('Student_grades.csv')

# Extract features and target variable
hours_studied = data['weekly_self_study_hours'].values.reshape(-1, 1)
grades = data['Final_Grade'].values

# Encode the target variable
le = LabelEncoder()
grades_encoded = le.fit_transform(grades)


X_train, X_test, y_train, y_test = train_test_split(hours_studied, grades_encoded, test_size=0.3, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(class_weight='balanced')
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

