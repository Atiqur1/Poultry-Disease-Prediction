import pandas as pd

# Sample dataset
data = {
    "Temperature": [30, 35, 28, 40, 33, 38, 29],
    "Humidity": [70, 85, 60, 90, 75, 80, 65],
    "Coughing": [1, 0, 1, 1, 0, 1, 0],
    "Diarrhea": [0, 1, 0, 1, 1, 0, 1],
    "Lethargy": [1, 1, 0, 1, 0, 1, 0],
    "Disease": ["Healthy", "Avian Flu", "Healthy", "Newcastle", "Healthy", "Newcastle", "Avian Flu"]
}

df = pd.DataFrame(data)
print(df.head())

# Data cleaning and preprocessing
from sklearn.preprocessing import LabelEncoder

# Check for missing values
print(df.isnull().sum())

# Encode categorical labels
le = LabelEncoder()
df["Disease"] = le.fit_transform(df["Disease"])  # Convert diseases to numbers

print(df.head())

#Exploratory Data Analysis

import seaborn as sns
import matplotlib.pyplot as plt

# Visualizing correlation
plt.figure(figsize=(8, 5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

#Application of Machine learning model

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Splitting data
X = df.drop(columns=["Disease"])
y = df["Disease"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


#Visualize result

# Feature importance visualization
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
feature_importance.sort_values().plot(kind="barh", color="skyblue")
plt.title("Feature Importance in Disease Prediction")
plt.xlabel("Importance Score")
plt.show()

#Deploy solutions

def predict_disease(temperature, humidity, coughing, diarrhea, lethargy):
    input_data = [[temperature, humidity, coughing, diarrhea, lethargy]]
    prediction = model.predict(input_data)
    return le.inverse_transform(prediction)[0]

# Example prediction
result = predict_disease(32, 80, 1, 0, 1)
print("Predicted Disease:", result)

