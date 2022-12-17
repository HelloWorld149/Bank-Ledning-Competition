import os
from Module.util import *
from Module.classifier import *
from Module.cluster import *

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_selector as selector


if __name__ == '__main__':
    os.system('cls' if os.name == 'nt' else 'clear')
    train, y, test, val_X, val_y = load_data(os.path.join(
        os.path.dirname(__file__), 'bank-lending-prediction\lending_train.csv'), os.path.join(os.path.dirname(__file__), 'bank-lending-prediction\lending_topredict.csv'))
    weight = clustering(train, test, y)
    #weight = pyliep_dist(train[['fico_score_range_high', 'fico_score_range_low']], test[['fico_score_range_high', 'fico_score_range_low']], y)
    #print(train.shape)
    run_classifier(train, y, test, weight, val_X, val_y)