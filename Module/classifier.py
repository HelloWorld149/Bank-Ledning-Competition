from calendar import c
import pandas as pd
import numpy as np
import math
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingRegressor
from collections import Counter

from sklearn.pipeline import make_pipeline

# 성과측정

from sklearn.metrics import roc_curve


from calendar import c
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import GradientBoostingClassifier
 

# 성과측정

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.ensemble import BaggingClassifier



def run_classifier(X, y, test_X, weight, val_X, val_y):

    acc = []
    #weight = np.array([4 if _y == 0 else 1 for _y in y])
    #X_train, X_val, y_train, y_val, weight_train, weight_val =  train_test_split(X, y, weight, test_size=0.2, stratify= y, random_state= 1)
    X_train, X_val, y_train, y_val, weight_train, weight_test =  train_test_split(X, y, weight, test_size=0.1, stratify= y, random_state= 1)
    print(X_train.shape)
    print(val_X.shape)
    #force_dist(X_val, y_val, test_X)
    submission_format = pd.read_csv('bank-lending-prediction\lending_topredict.csv').loc[:, ['ID', 'loan_paid']]
    #max depth = 7 has highest validation accuracy
    clf = DecisionTreeClassifier(max_depth= 7)
    clf_1 = DecisionTreeClassifier(max_depth= 13)
    print("Classifier 1 110 7 430000 no weight")
    
    #adaboost = Classfier(clf, 20)
    max_score = 0
    max_idx = 0 
    xgboost = GradientBoostingRegressor(n_estimators = 110, max_depth= 7, learning_rate = 0.1)
    #xgboost = GradientBoostingClassifier()
    #xgboost_1 = GradientBoostingRegressor(n_estimators = 110, max_depth= 7, learning_rate = 0.1)
    #adaboost = AdaBoostClassifier(base_estimator=clf, n_estimators=20)
    #bagging = BaggingClassifier(base_estimator=clf_1, n_estimators=20)
    xgboost.fit(X_train, y_train)
    #adaboost.fit(X_train, y_train)
    #bagging.fit(X_train, y_train)
    y_train_pred = np.array(xgboost.predict(X_train))
    y_val_pred = np.array(xgboost.predict(val_X))
    #y_train_pred_ada = adaboost.predict_proba(X_train)[:,1]
    #y_val_pred_ada = adaboost.predict_proba(X_val)[:,1]
    #y_train_pred_bag = bagging.predict_proba(X_train)[:,1]
    #y_val_pred_bag = bagging.predict_proba(X_val)[:,1]

    #xg_w = 0.6439563791268181
    #ada_w = 0.6379486058754571
    #bag_w = 0.634625167887938
    #sum = xg_w + ada_w + bag_w
    #xg_w, ada_w, bag_w = xg_w / sum, ada_w / sum, bag_w / sum
    
    #y_train_pred = xg_w * y_train_pred_xg + ada_w * y_train_pred_ada + bag_w * y_train_pred_bag
    #y_val_pred = xg_w * y_val_pred_xg + ada_w * y_val_pred_ada + bag_w * y_val_pred_bag
    print("Classifier 2")

    #threshold = Find_Optimal_Cutoff(y_train, y_train_pred)
    #y_train_pred = np.array(y_train_pred > threshold).astype(int)  
    #y_val_pred = np.array(y_val_pred > threshold).astype(int) 
    threshold, best_mcc, y_train_pred = eval_mcc(np.array(y_train), y_train_pred)
    y_val_pred = np.array(y_val_pred > threshold).astype(int) 
    print(y_train_pred)
    print(y_val_pred)

    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    val_accuracy = accuracy_score(val_y, y_val_pred)
    train_auc_score = roc_auc_score(y_train, y_train_pred)
    val_auc_score = roc_auc_score(val_y, y_val_pred)
    matthe_train = matthews_corrcoef(y_train, y_train_pred)
    matthe_val = matthews_corrcoef(val_y, y_val_pred)
    acc.append(train_accuracy)
    acc.append(val_accuracy)
    acc.append(train_auc_score)
    acc.append(val_auc_score)
    acc.append(matthe_train)
    acc.append(matthe_val)
    
    print("balance 7 110_mcc_opt0.9")
    print(train_accuracy)
    print(val_accuracy) 
    print(train_auc_score)
    print(val_auc_score)   
    print(max_idx)
    print(max_score)
    print(acc)
    print(matthe_train)
    print(matthe_val)

    y_test_pred = xgboost.predict(test_X)
    #y_test_pred_ada = adaboost.predict_proba(test_X)[:,1]    
    #y_test_pred_bag = bagging.predict_proba(test_X)[:,1]
    #y_test_pred = xg_w * y_test_pred_xg + ada_w * y_test_pred_ada + bag_w * y_test_pred_bag
    y_test_pred = np.array(y_test_pred > threshold).astype(int)  
    submission_format['loan_paid'] = y_test_pred
    submission_format.to_csv("result/balance 7 110_mcc_opt0.9_430000.csv", index=False)
    

#target = target class 
#predicted = predicted class
#return value: the list of the optimal cutoff
def Find_Optimal_Cutoff(target, predicted):
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index = i), 'threshold' : pd.Series(threshold, index = i)})
    
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
    return list(roc_t['threshold'])



def mcc(tp, tn, fp, fn):
    sup = tp * tn - fp * fn
    inf = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if inf==0:
        return 0
    else:
        return sup / np.sqrt(inf)


def eval_mcc(y_true, y_prob):
    idx = np.argsort(y_prob)
    y_true_sort = y_true[idx]
    n = y_true.shape[0]
    nump = 1.0 * np.sum(y_true) # number of positive
    numn = n - nump # number of negative
    tp = nump
    tn = 0.0
    fp = numn
    fn = 0.0
    best_mcc = 0.0
    best_id = -1
    prev_proba = -1
    best_proba = -1
    mccs = np.zeros(n)
    for i in range(n):
        # all items with idx < i are predicted negative while others are predicted positive
        # only evaluate mcc when probability changes
        proba = y_prob[idx[i]]
        if proba != prev_proba:
            prev_proba = proba
            new_mcc = mcc(tp, tn, fp, fn)
            if new_mcc >= best_mcc:
                best_mcc = new_mcc
                best_id = i
                best_proba = proba
        mccs[i] = new_mcc
        if y_true_sort[i] == 1:
            tp -= 1.0
            fn += 1.0
        else:
            fp -= 1.0
            tn += 1.0

    y_pred = np.array(y_prob >= best_proba).astype(int)
    score = matthews_corrcoef(y_true, y_pred)
    print(score, best_mcc)
    return best_proba, best_mcc, y_pred
