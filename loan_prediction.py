import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("../train_u6lujuX_CVtuZ9i.csv")
df.dropna(inplace=True)

df = df.drop("Loan_ID", axis=1)
df["LoanAmount"].fillna(df["LoanAmount"].median(), inplace=True)
df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].median(), inplace=True)
df["Credit_History"].fillna(df["Credit_History"].mode()[0], inplace=True)
df["Dependents"] = df["Dependents"].replace("3+", "3")
df["Dependents"] = df["Dependents"].astype(int)
for col in ["Gender", "Married", "Self_Employed", "Dependents"]:
    df[col].fillna(df[col].mode()[0], inplace=True)

# ---- Binary Encoding ----
df["Gender"] = df['Gender'].map({"Male":1, "Female":0})
df["Education"] = df['Education'].map({"Graduate":1, "Not Graduate":0})
df["Married"] = df['Married'].map({"Yes":1, "No":0})
df["Self_Employed"] = df['Self_Employed'].map({"Yes":1, "No":0})
df["Loan_Status"] = df['Loan_Status'].map({"Y":1, "N":0})

# ---- ONE HOT for Property Area ----
df = pd.get_dummies(df, columns=["Property_Area"], drop_first=True)
# for col in cat_cols:
#     print(df[col].value_counts())
# df = df.drop(["Dependents", "Married", "ApplicantIncome"], axis=1)
# sns.countplot(x="Credit_History", data=df)
# df["Credit_History"].hist()
# sns.countplot(x='Credit_History', hue='Loan_Status', data=df)
# sns.heatmap(df.select_dtypes(include=['int64','float64']).corr(), annot=True)
# plt.title("Correlation Heatmap")
# plt.show()

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
#Model used of logistic Regression

# df[num_cols] = scale.fit_transform(df[num_cols])
x = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

num_cols = x.select_dtypes(include=['int64','float64']).columns
scale = StandardScaler()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
x_train[num_cols] = scale.fit_transform(x_train[num_cols])
x_test[num_cols] = scale.transform(x_test[num_cols])


# model = LogisticRegression(class_weight='balanced', max_iter=500)#logistic regression model
# model = GaussianNB()#naive bayes model
model = SVC(kernel='linear', class_weight='balanced', C=1.0)
#till now svc is better than naive bayes and logistic regression model
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
# print(y_pred)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score

print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("F1 Score", f1_score(y_test, y_pred))