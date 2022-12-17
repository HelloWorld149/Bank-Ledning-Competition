import pandas as pd
import numpy as np
from Module.classifier import *
import os

#to mix categorical and numerical
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_selector as selector

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef

#read the file name
def read_data(file_name):
    dataset = pd.read_csv(file_name)
    
    #abaondon this column since > 50% NA
    dataset = dataset.drop(columns = ["months_since_last_delinq"])
    
    def _label_encoding():
        cols = ["loan_duration", "employment", "employment_length", "reason_for_loan",
             "extended_reason", "employment_verified", "zipcode", "state", "home_ownership_status", "type_of_application"]
        label_dict = {}
        ## >>> YOUR CODE HERE >>>
        for col in cols:
            column_np = np.array(dataset.loc[:, col])
            column_np = np.unique(column_np) 
            
            column_encode = np.arange(column_np.shape[0])
            encode = 0
            for label in column_np:
                label_dict[col, label] = encode
                encode += 1     
            dataset[col] = dataset[col].replace(column_np, column_encode)
    return dataset

#analyze the data
#result 4:1 ratio imbalanced.
def check_data(file_name):
    #read the data
    lending_dataset = read_data(file_name)
    #dataset_card_fraud.drop(['Time'], axis=1, inplace=True)
    lending_dataset.head()
    X = lending_dataset.iloc[:,1:23]
    print(X.head())
    y = lending_dataset.iloc[:,24]
    print(y.head())

    #dataset_card_fraud = pd.read_csv('resample_rd_10000')
    #X = dataset_card_fraud.iloc[:,:29]
    #print(X.head())
    #y = dataset_card_fraud.iloc[:,29]
    #print(y.head())
    print('paid: ', round(lending_dataset['loan_paid'].value_counts()[1]/len(lending_dataset) * 100))
    print('not paid: ', round(lending_dataset['loan_paid'].value_counts()[0]/len(lending_dataset) * 100))
    

def run_classifier():
    dataset = read_data("bank-lending-prediction\lending_train.csv")
    test = read_data("bank-lending-prediction\lending_topredict.csv")
    #create X and y
    X = dataset.iloc[:,1:23]
    print(X.shape)
    y = dataset.iloc[:,23]
    print(y.shape)

    submission = test
    test_X = test.iloc[:,1:23]
    submission_format = submission.loc[:, ['ID', 'loan_paid']]
    print(submission_format.columns)
    print(submission_format)
    
    numeric_features = ["requested_amnt", "annual_income", "debt_to_income_ratio", "fico_score_range_low",
               "fico_score_range_high", "revolving_balance", "total_revolving_limit"]
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_features = ["loan_duration", "employment", "employment_length", "race", "reason_for_loan",
             "extended_reason", "employment_verified", "public_bankruptcies","zipcode", "state", 
             "public_bankruptcies", "zipcode", "state", "home_ownership_status", 
             "delinquency_last_2yrs","type_of_application",
             "fico_inquired_last_6mths", "any_tax_liens"
             ]
    
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    X_train, X_val, y_train, y_val =  train_test_split(X, y, test_size=0.2, stratify= y, random_state= 1)
    print(X_train.shape)
    print(X_val.shape)
    acc = []

    #max depth = 7 has highest validation accuracy
    clf = DecisionTreeClassifier(max_depth= 2)
    
    #adaboost = Classfier(clf, 20)
    max_score = 0
    max_idx = 0 
    adaboost = Pipeline(
    steps=[("preprocessor", preprocessor), 
        ("classifier", AdaBoostClassifier(base_estimator= clf, n_estimators=20))]
    )
    adaboost.fit(X_train, y_train)
    y_train_pred = adaboost.predict_proba(X_train)
    print(y_train_pred)
    y_val_pred = adaboost.predict_proba(X_val)

    threshold = Find_Optimal_Cutoff(y_train, y_train_pred[:, 1])
    y_train_pred = np.array(y_train_pred[:, 1] > threshold).astype(int).flatten()   
  
    y_val_pred = np.array(y_val_pred[:, 1] > threshold).astype(int).flatten()   
    
    print(y_train.shape)
    print(y_train_pred.shape)
    print(y_val.shape)
    print(y_val_pred.shape)
    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    val_accuracy = accuracy_score(y_val_pred, y_val)
    train_auc_score = roc_auc_score(y_train, y_train_pred)
    val_auc_score = roc_auc_score(y_val_pred, y_val)
    acc.append(train_accuracy)
    acc.append(val_accuracy)
    acc.append(train_auc_score)
    acc.append(val_auc_score)
    
    print(train_accuracy)
    print(val_accuracy) 
    print(train_auc_score)
    print(val_auc_score)   
    print(max_idx)
    print(max_score)
    print(acc)

    y_test_pred = adaboost.predict_proba(test_X)
    y_test_pred = np.array(y_test_pred[:, 1] > threshold).astype(int).flatten()   
    submission_format['loan_paid'] = y_test_pred
    submission_format.to_csv("result/decision_tree_dep=2.csv", index=False)
        
        
if __name__ == '__main__':
    os.system('cls' if os.name == 'nt' else 'clear')
    run_classifier()
    

#target = target class 
#predicted = predicted class
#return value: the list of the optimal cutoff
def Find_Optimal_Cutoff(target, predicted):
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index = i), 'threshold' : pd.Series(threshold, index = i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
    return list(roc_t['threshold'])

