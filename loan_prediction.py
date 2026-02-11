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
from matplotlib.colors import ListedColormap
#Model used of logistic Regression

# df[num_cols] = scale.fit_transform(df[num_cols])
x = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

num_cols = x.select_dtypes(include=['int64','float64']).columns
scale = StandardScaler()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
x_train[num_cols] = scale.fit_transform(x_train[num_cols])
x_test[num_cols] = scale.transform(x_test[num_cols])

zero_one_colourmap = ListedColormap(('blue', 'red'))
def plot_decision_boundary(X, y, clf):
    X_set, y_set = X.values, y.values
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, 
                                 stop = X_set[:, 0].max() + 1, 
                                 step = 0.01),
                       np.arange(start = X_set[:, 1].min() - 1, 
                                 stop = X_set[:, 1].max() + 1, 
                                 step = 0.01))
  
    plt.contourf(X1, X2, clf.predict(np.array([X1.ravel(), 
                                             X2.ravel()]).T).reshape(X1.shape),
               alpha = 0.75, 
               cmap = zero_one_colourmap)
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = (zero_one_colourmap)(i), label = j)
    plt.title('SVM Decision Boundary')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    return plt.show()

def plot_3d_plot(X, y):
    X = np.array(X)
    r = np.exp(-(X ** 2).sum(axis=1))
    # fig = plt.figure(figsize=(8, 6))
    ax = plt.subplot(projection='3d')
    ax.scatter3D(X[:, 0], X[:, 1], r, c=y, s=100, cmap='bwr')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('y')
    plt.show()
    return ax

features = ["Credit_History", "LoanAmount"]
x_vis = x_train[features]
x_test_vis = x_test[features]

# model = LogisticRegression(class_weight='balanced', max_iter=500)#logistic regression model
# model = GaussianNB()#naive bayes model
model = SVC(kernel='rbf', class_weight='balanced', C=1.0)
#till now svc is better than naive bayes and logistic regression model
model.fit(x_vis, y_train)

# plot_decision_boundary(x_vis, y_train, model)
plot_3d_plot(x_vis,y_train)
y_pred = model.predict(x_test_vis)
# print(y_pred)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score

print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("F1 Score", f1_score(y_test, y_pred))