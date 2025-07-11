# Import Necessary Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from sklearn.datasets import load_iris

# Load the Dataset
iris = load_iris()

# Create a Dataframe
df = pd.DataFrame(data=iris.data,columns=iris.feature_names)
df['target'] = iris.target
df['flower_name'] = df['target'].apply(lambda x: iris.target_names[x])

# Display the iris 5 rows
print("Dataset Preview:")
print(df.head())

# Data Visualization
sns.pairplot(df,hue='flower_name')
plt.suptitle("Iris Flower Pair Plot",y=1.02)
plt.show()

# Features and Labels
X = df[iris.feature_names]
y = df['target']

# Train and SPlit the
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Create and train Logistic Regression
model = LogisticRegression(max_iter=200)
model.fit(X_train,y_train)

# Prediction on the test Data
y_pred = model.predict(X_test)

# Evaluation
print("\nConfusion Matrix:")
print(confusion_matrix(y_test,y_pred))

print("\nClassification Report:")
print(classification_report(y_test,y_pred,target_names=iris.target_names))

print("\nAccuracy Score:",accuracy_score(y_test,y_pred))

# Test with a new Sample
sample = [[5.1,3.5,1.4,0.2]]
predicted_class = model.predict(sample)
print("\nSample Prediction:",iris.target_names[predicted_class[0]])

# Output
# Dataset Preview:
#    sepal length (cm)  sepal width (cm)  ...  target  flower_name
# 0                5.1               3.5  ...       0       setosa
# 1                4.9               3.0  ...       0       setosa
# 2                4.7               3.2  ...       0       setosa
# 3                4.6               3.1  ...       0       setosa
# 4                5.0               3.6  ...       0       setosa

# [5 rows x 6 columns]

# Confusion Matrix:
# [[10  0  0]
#  [ 0  9  0]
#  [ 0  0 11]]

# Classification Report:
#               precision    recall  f1-score   support

#       setosa       1.00      1.00      1.00        10
#   versicolor       1.00      1.00      1.00         9
#    virginica       1.00      1.00      1.00        11

#     accuracy                           1.00        30
#    macro avg       1.00      1.00      1.00        30
# weighted avg       1.00      1.00      1.00        30


# Accuracy Score: 1.0

# Sample Prediction: setosa
