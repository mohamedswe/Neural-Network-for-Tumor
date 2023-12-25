import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load the dataset
df = pd.read_csv("C://Users//mabde//OneDrive//Desktop//Coding//data sciene//heart_failure_clinical_records_dataset.csv")

# Split the data into features (X) and target (y)
X = df.drop(columns=["DEATH_EVENT"])
y = df["DEATH_EVENT"]

# Standardize the data to have mean=0 and variance=1, which is generally good for logistic regression
scaler = StandardScaler()
X_normal = scaler.fit_transform(X)

# Initialize the logistic regression model
logistic_model = LogisticRegression()

# Use 10-fold cross validation
scores = cross_val_score(logistic_model, X_normal, y, cv=10, scoring='accuracy')

print(f"Mean Accuracy of Logistic Regression with 10-fold CV: {scores.mean()*100:.2f}%")

