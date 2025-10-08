---
title: "Regression Practice with Titanic Dataset"
date: "2025-10-08"
description: "A beginner-friendly walkthrough of logistic regression using the Titanic dataset"
tags: ["regression", "logistic regression", "machine learning", "titanic"]
---

## 📊 Introduction

This post walks through a practical example of logistic regression using the Titanic dataset. We explore data preprocessing, model training, evaluation, and feature importance.

---## 🧼 Data Preparation
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np from sklearn.linear_model 
import LogisticRegressionfrom sklearn.model_selection 
import train_test_splitfrom sklearn.metrics 
import accuracy_score, confusion_matrix

# Load the datasetdata = pd.read_csv('train.csv')

data['Age'] = data.groupby(['Sex', 'Pclass'], group_keys=False)['Age'].apply(lambda x: x.fillna(x.mean()))

# Visualizing Age Imputation
fig, axes = plt.subplots(1, 2, figsize=(6, 8), sharey=True)
sns.histplot(data_original['Age'], ax=axes[0], kde=True, color='red')
sns.histplot(data['Age'], ax=axes[1], kde=True, color='green')
plt.show()

# Converting Categorical Variables 
data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)

# Building the Logistic Regression Model
X = data[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']]
y = data['Survived']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train Model 
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Evaluate Model 
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\\n{conf_matrix}')

# Feature Importance 
feature_importance = model.coef_[0]
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)

sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance')
plt.show()

