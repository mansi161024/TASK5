# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn import tree

# Load dataset
df = pd.read_csv(r"C:\Users\acer\Desktop\AI&ML\day5.py\8c7d615b96f6eeaa95f00843e3820287.csv")  # Adjusted path for uploaded file
print(df.head())
print(df.info())

# Preprocessing
df = df.dropna()  # Simple method: drop rows with missing values
X = df.drop('target', axis=1)  # Replace 'target' with your actual target column
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Train Decision Tree
dtree = DecisionTreeClassifier(max_depth=4, random_state=42)
dtree.fit(X_train, y_train)

# Evaluate Decision Tree
y_pred_tree = dtree.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_tree))
print(classification_report(y_test, y_pred_tree))

# Visualize Decision Tree
plt.figure(figsize=(20,10))
plot_tree(dtree, filled=True, feature_names=X.columns, class_names=True)
plt.title("Decision Tree Visualization")
plt.show()

# 2. Train Random Forest
rforest = RandomForestClassifier(n_estimators=100, random_state=42)
rforest.fit(X_train, y_train)

# Evaluate Random Forest
y_pred_rf = rforest.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# 3. Feature Importance
importances = rforest.feature_importances_
feature_names = X.columns
forest_importances = pd.Series(importances, index=feature_names)
forest_importances.sort_values().plot(kind='barh', figsize=(10, 6))
plt.title("Feature Importances from Random Forest")
plt.show()

# 4. Cross-validation scores
dtree_scores = cross_val_score(dtree, X, y, cv=5)
rforest_scores = cross_val_score(rforest, X, y, cv=5)

print("Decision Tree CV Accuracy:", dtree_scores.mean())
print("Random Forest CV Accuracy:", rforest_scores.mean())