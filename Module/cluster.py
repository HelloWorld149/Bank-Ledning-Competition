import seaborn as sns; sns.set()
from collections import Counter

from sklearn import mixture
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn import datasets
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from Module.pykliep import DensityRatioEstimator

def pyliep_dist(X, test, y):
    X_train, X_val =  train_test_split(X, test_size=0.9, stratify= y, random_state= 1)
    kliep = DensityRatioEstimator()
    kliep.fit(np.array(X_train), np.array(test)) # keyword arguments are X_train and X_test
    weights = kliep.predict(np.array(X))
    return weights


def clustering(X, test, y):
    print("start clustering")
    print(X.shape)
    # fit a Gaussian Mixture Model with four components
    X_train, X_val =  train_test_split(X, test_size=0.3, random_state= 1)
    print(X_train)
    
    KMean= KMeans(n_clusters=2)
    KMean.fit(X_train)
    labels_val = KMean.predict(X_val)
    labels_test = KMean.predict(test)
    labels = np.array(KMean.predict(X))
    
    zero_weight = 0
    one_weight = 0
    w = []
    for val in [labels_val, labels_test, labels]:
        counts = Counter()
        for count in val:
            counts[count] += 1
        print([(i, counts[i] / len(val) * 100.0) for i in counts])
        w.append([counts[0] / len(val), counts[1] / len(val)])
    #print(metrics.silhouette_score(X_val, labels_val, metric = 'euclidean'))
    N = labels.shape[0]
    print(N)
    print(w)

    weight_zero = w[1][0] / w[0][0]
    weight_one = w[1][1] / w[0][1]
    #weight_zero = w[0][0] / w[1][0]
    #weight_one = w[0][1] / w[1][1]
    print(weight_zero)
    print(weight_one)
    
    check = [True if (x == 0 and _y == 1) else False for x, _y in zip(labels, y)]
    check_1 = [True if (x == 1 and _y == 0) else False for x, _y in zip(labels, y)]
    check_idx = [True if x == 0 else False for x in labels]
    check_1_idx = [True if x == 1 else False for x in labels]
    
    #check_df = X.iloc[check_idx, :]
    #check_1_df = X.iloc[check_1_idx, :]
    #y_df = y.iloc[check_idx]
    #print("===============")
    #print(y_df.value_counts()[1])
    #print(y_df.value_counts()[0])
    #print(X['fico_score_range_high'].describe())
    #print(check_df['fico_score_range_high'].describe())
    #print(check_1_df['fico_score_range_high'].describe())
    #print(test['fico_score_range_high'].describe())
    #print("==================")

    print(y.value_counts()[1])
    print(y.value_counts()[0])
    print(labels)
    print(y)
    print(check.count(1))
    print(check.count(0))
    print(check_1.count(1))
    print(check_1.count(0))
    
    
    weight = np.array([weight_zero if x == 0 else weight_one for x in labels])
    print(labels)
    print(weight)
    
    return weight
    
    
    
        
    
    
        

            


