import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline 
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,precision_score,recall_score
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier


# Loading dataset
df = pd.read_csv(r"c:\Users\vivek\Downloads\diabetes (1).csv")

# Data inspection
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())
print(df["Outcome"].value_counts())

#there is no missing value in the data

#detecting outlier
sns.boxplot(data=df)
plt.show()

for col in df.drop(columns="Outcome").columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    
    IQR = Q3 - Q1
    
    outliers = ((df[col] < Q1 - 1.5 * IQR) | 
                (df[col] > Q3 + 1.5 * IQR)).sum()
    
    print(col, ":", outliers)
    
#fixing outlier
for col in ["Insulin", "SkinThickness"]:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df[col] = df[col].clip(lower, upper)    

sns.boxplot(data=df)
plt.show()

# Preprocessing()
X = df.drop(['Outcome'], axis=1)
y = df['Outcome']
 
#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#using Smote
smote= SMOTE(random_state=0)
X_train_smote,y_train_smote=smote.fit_resample(X_train,y_train)

# Define models
models = {
    'LogisticRegression': Pipeline([("scaler",StandardScaler()),("model",LogisticRegression())]),
    'KNN': Pipeline([("scaler",StandardScaler()),("model",KNeighborsClassifier())]),
    'SVM': Pipeline([("scaler",StandardScaler()),("model",SVC())]),
    'NaiveBayes': Pipeline([("scaler",StandardScaler()),("model",GaussianNB())]),
    'DecisionTree': Pipeline([("scaler",StandardScaler()),("model",DecisionTreeClassifier())]),
    'RandomForest': Pipeline([("scaler",StandardScaler()),("model",RandomForestClassifier())]),
    'XGBoost':  Pipeline([("scaler",StandardScaler()),("model",xgb.XGBClassifier())]),
}

# Train and evaluate models
results = []

for name, model in models.items():
    model.fit(X_train_smote, y_train_smote)
    y_pred = model.predict(X_test)
    
    cm= confusion_matrix(y_test, y_pred)
    acs= accuracy_score(y_test, y_pred)
    ps=precision_score(y_test, y_pred)
    rc=recall_score(y_test, y_pred)
    f1=f1_score(y_test, y_pred)
    
    results.append({
        'Model': name,
        'Confusion_matrix': cm,
        'Accuracy': acs,
        'Precision': ps,
        'Recall': rc,
        'F1 score': f1
    })

# Convert results to DataFrame and save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('diabetes_model_evaluation_results.csv', index=False)

#Hyperparameter tuning

#hyperparameter tuning of Random_Forest model
rf = RandomForestClassifier(random_state=0)

rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_grid = GridSearchCV(
    estimator=rf,
    param_grid=rf_params,
    cv=5,
    scoring='recall',
    n_jobs=-1
)

rf_grid.fit(X_train_smote, y_train_smote)

print("Best Random Forest Parameters:")
print(rf_grid.best_params_)

print("Best CV Recall Score:")
print(rf_grid.best_score_)

# Test the best model
rf_best = rf_grid.best_estimator_

y_pred_rf = rf_best.predict(X_test)

print("Test Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Test Precision:", precision_score(y_test, y_pred_rf))
print("Test Recall:", recall_score(y_test, y_pred_rf))
print("Test F1 Score:", f1_score(y_test, y_pred_rf))
print("Confusion Matrix:",confusion_matrix(y_test, y_pred_rf))

#hyperparameter tuning of XG_Boost model

xgb_model = xgb.XGBClassifier(
    random_state=0
)

xgb_params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}

xgb_grid = GridSearchCV(
    estimator=xgb_model,
    param_grid=xgb_params,
    cv=5,
    scoring='recall',
    n_jobs=-1
)

xgb_grid.fit(X_train_smote, y_train_smote)

print("Best XGBoost Parameters:")
print(xgb_grid.best_params_)

print("Best CV recall Score:")
print(xgb_grid.best_score_)

# Test the best model
xgb_best = xgb_grid.best_estimator_

y_pred_xgb = xgb_best.predict(X_test)

print("Test Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("Test Precision:", precision_score(y_test, y_pred_xgb))
print("Test Recall:", recall_score(y_test, y_pred_xgb))
print("Test F1 Score:", f1_score(y_test, y_pred_xgb))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_xgb))

#using pickle


with open("diabetes_model.pkl", "wb") as file:
    pickle.dump(xgb_best, file)

print("Model saved successfully!")

